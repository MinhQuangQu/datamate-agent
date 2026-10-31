"""DataMate Agent entry point (05.8).

    python -m dma.run --config configs/final.yaml [--data /data] [--result /result] [--profile accuracy|fast]
        [--limit 50] [--resume] [--no-llm-start --endpoints http://127.0.0.1:8001/v1,...] [--warmup-only]
        [--only-business <id>,<id>]   # local debugging only — predict.sh never passes it (Rule 4)
        [--fake-llm]                        # test double, dev/CI only

Always leaves a valid /result/submission.jsonl with every id (abstain for unfinished ones)
and exits 0 (Rule 8: only written lines are scored).
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import platform
import signal
import sys
import threading
import time
import traceback
from datetime import datetime, timezone

from dma.config import Cfg, apply_profile, asset_path, config_sha256, load_config
from dma.context import SplitContext, load_calendar
from dma.io.split import discover_split, load_questions, sha256_file
from dma.llm.gateway import LLMGateway, content_seed
from dma.memory.exemplars import ExemplarMemory
from dma.output.formatter import FormatConfig
from dma.output.writer import SubmissionWriter, write_log
from dma.rules.extract import build_split_rulespec, load_reference
from dma.semantic.bkp import BusinessKnowledgePack, finalize_business, minimal_bkp, prepare_business
from dma.semantic.bsm import BSM
from dma.semantic.builder import refine_bsm, repair_status_maps
from dma.semantic.catalog import load_weekday_names
from dma.types import Question
from dma.verify.calibrator import Calibrator

VERSION = "1.0.0"


class Events:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, kind: str, data: dict | None = None) -> None:
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": kind, **(data or {})}
        line = json.dumps(rec, ensure_ascii=False, default=str)
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(f"[dma] {kind} {json.dumps(data or {}, ensure_ascii=False, default=str)[:300]}", flush=True)


class Clock:
    """Wall-clock budget from predict.sh start (DMA_START_EPOCH) — only for emergency degradation."""

    def __init__(self, soft: float, hard: float):
        self.t0 = float(os.environ.get("DMA_START_EPOCH") or time.time())
        self.soft, self.hard = soft, hard

    def elapsed(self) -> float:
        return time.time() - self.t0

    def past_soft(self) -> bool:
        return self.elapsed() > self.soft

    def past_hard(self) -> bool:
        return self.elapsed() > self.hard


def assign_replicas(by_biz: dict[str, list], n: int) -> dict[str, int]:
    """Sticky routing balanced by question count (deterministic: largest first, then id)."""
    load = [0] * max(1, n)
    out = {}
    for biz in sorted(by_biz, key=lambda b: (-len(by_biz[b]), b)):
        r = min(range(len(load)), key=lambda i: (load[i], i))
        out[biz] = r
        load[r] += len(by_biz[biz])
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="dma.run")
    ap.add_argument("--config", default=os.environ.get("DMA_CONFIG"))
    ap.add_argument("--data")
    ap.add_argument("--result")
    ap.add_argument("--work")
    ap.add_argument("--profile")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only-business")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-llm-start", action="store_true")
    ap.add_argument("--endpoints")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--warmup-only", action="store_true")
    ap.add_argument("--exclude-memory-businesses", default="")
    ap.add_argument("--concurrency", type=int)
    return ap.parse_args(argv)


def build_cfg(a: argparse.Namespace) -> Cfg:
    over: dict = {"run": {}, "llm": {}}
    if a.data:
        over["run"]["data_root"] = a.data
    if a.result:
        over["run"]["result_dir"] = a.result
    if a.work:
        over["run"]["work_dir"] = a.work
    if a.resume:
        over["run"]["resume"] = True
    if a.concurrency:
        over["run"]["max_concurrency"] = a.concurrency
    if a.no_llm_start or a.endpoints:
        over["llm"]["start_servers"] = False
    if a.endpoints:
        over["llm"]["endpoints"] = [e.strip() for e in a.endpoints.split(",") if e.strip()]
    if a.no_llm:
        over["llm"]["enabled"] = False
    if a.fake_llm:
        over["llm"]["backend"] = "fake"
        over["llm"]["start_servers"] = False
    cfg = load_config(a.config, over)
    return apply_profile(cfg, a.profile or cfg.run.profile)


class Runner:
    def __init__(self, cfg: Cfg, args: argparse.Namespace):
        self.cfg, self.args = cfg, args
        self.result_dir = cfg.run.result_dir
        self.log_dir = os.path.join(self.result_dir, "logs")
        self.work_dir = cfg.run.work_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        self.events = Events(os.path.join(self.log_dir, "events.jsonl"))
        self.clock = Clock(float(cfg.run.soft_deadline_s), float(cfg.run.hard_deadline_s))
        self.writer: SubmissionWriter | None = None
        self.supervisor = None
        self.llm: LLMGateway | None = None
        self.pool = None
        self.all_ids: list[str] = []
        self.stop = threading.Event()
        self.manifest: dict = {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                               "version": VERSION, "profile": cfg.run.profile, "config_sha256": config_sha256(cfg),
                               "seed_base": cfg.llm.seed_base, "businesses": [], "warnings": []}
        self.stats = {"answered": 0, "abstain": 0, "unanswerable": 0, "errors": 0, "done": 0}
        self.durations: list[float] = []

    # ------------------------------------------------------------------ setup
    def start_llm(self) -> None:
        lc = self.cfg.llm
        if not lc.enabled:
            self.llm = LLMGateway(self.cfg, endpoints=[])
            self.events("llm_disabled", {})
            return
        if lc.backend == "fake":
            from dma.llm.fake import FakeBackend
            self.llm = LLMGateway(self.cfg, endpoints=[], backend=FakeBackend())
            self.events("llm_fake_backend", {"warning": "test double: not a model"})
            return
        if lc.start_servers:
            from dma.llm.supervisor import Supervisor
            self.supervisor = Supervisor(self.cfg, self.log_dir, on_event=self.events)
            try:
                eps = self.supervisor.start()
            except OSError as e:          # vllm binary / model path missing: keep going, everything abstains
                self.events("vllm_start_failed", {"error": repr(e)[:300]})
                eps = []
        else:
            eps = list(lc.endpoints)
        self.llm = LLMGateway(self.cfg, endpoints=eps)
        self.llm.on_event = self.events

    async def wait_llm(self) -> None:
        if self.llm is None or not self.llm.enabled or self.llm.backend is not None:
            return
        timeout = float(self.cfg.llm.startup_timeout_s)
        if self.supervisor is not None:
            ready = await asyncio.to_thread(self.supervisor.wait_ready, timeout)
        else:
            t0 = time.monotonic()
            first_ok = None
            ready = await self.llm.health()
            while not all(ready) and time.monotonic() - t0 < min(timeout, 120):
                if any(ready):
                    first_ok = first_ok or time.monotonic()
                    if time.monotonic() - first_ok > 15:      # others get 15 s more, then we go on without them
                        break
                await asyncio.sleep(3)
                ready = await self.llm.health()
        for i, ok in enumerate(ready):
            if not ok:
                self.llm.mark_unhealthy(i, "not ready at startup")
        if not any(ready):
            self.llm.enabled = False
            self.events("llm_unavailable", {"note": "no replica ready: rule-only answers, mostly abstain"})
            return
        for i, ok in enumerate(ready):     # warmup: one tiny request per replica
            if ok:
                try:
                    await self.llm.chat("warmup", [{"role": "user", "content": "Trả lời: OK"}], seed=1, replica=i,
                                        max_tokens=8, thinking=False)
                except Exception as e:  # noqa: BLE001
                    self.events("warmup_failed", {"replica": i, "error": repr(e)[:300]})
        if self.supervisor is not None:
            self.supervisor.watch(self.llm)
        self.events("llm_ready", {"replicas": ready})

    def split_context(self, split) -> tuple[SplitContext, dict]:
        cfg = self.cfg
        ref_items, ref_spec = load_reference(asset_path(cfg, cfg.rules.reference_rulebook),
                                             asset_path(cfg, cfg.rules.reference_rulespec))
        rb = open(split.rulebook_path, encoding="utf-8-sig").read() if split.rulebook_path else None
        cal, cwarn = load_calendar(split.calendar_path)
        for w in cwarn:
            self.events("calendar_warning", {"warning": w})
        fmt = FormatConfig(rounding=cfg.format.rounding, max_list_len=int(cfg.format.max_list_len),
                           max_table_rows=int(cfg.format.max_table_rows))
        wd = load_weekday_names(asset_path(cfg, cfg.format.weekday_names))
        return SplitContext.make(ref_spec, cal, wd, fmt, rb or "", {}, split.calendar_path), \
            {"rb": rb, "ref_items": ref_items, "ref_spec": ref_spec}

    # ------------------------------------------------------------------ main
    async def main(self) -> None:
        cfg, a = self.cfg, self.args
        split = discover_split(cfg.run.data_root)
        for w in split.warnings:
            self.events("split_warning", {"warning": w})
        questions, qerr = load_questions(split.questions_path)
        for e in qerr[:50]:
            self.events("question_parse_error", {"error": e})
        self.all_ids = [q.id for q in questions]
        if a.only_business:
            keep = set(a.only_business.split(","))
            questions = [q for q in questions if q.business_id in keep]
        if a.limit:
            questions = questions[: a.limit]
        self.writer = SubmissionWriter(os.path.join(self.result_dir, "submission.jsonl"), resume=bool(cfg.run.resume))
        todo = [q for q in questions if q.id not in self.writer.done]
        self.manifest.update(split_root=split.root, n_questions=len(self.all_ids),
                             rulebook_sha256=sha256_file(split.rulebook_path), calendar_sha256=sha256_file(split.calendar_path),
                             env=_env())
        self.events("start", {"questions": len(self.all_ids), "todo": len(todo), "businesses": len(split.businesses),
                              "profile": cfg.run.profile})
        self.start_llm()
        if a.warmup_only:
            await self.wait_llm()
            return
        by_biz: dict[str, list[Question]] = {}
        for q in sorted(todo, key=lambda q: (q.business_id, q.id)):
            by_biz.setdefault(q.business_id, []).append(q)
        # CPU phase 1 for every business immediately (overlaps vLLM loading)
        n_workers = int(cfg.run.ingest_workers) or max(1, min(16, (os.cpu_count() or 4) - 2))
        ex = cf.ProcessPoolExecutor(max_workers=min(n_workers, max(1, len(by_biz))), mp_context=mp.get_context("spawn"))
        futs = {}
        for biz in by_biz:
            if biz not in split.businesses:
                self.events("business_missing", {"business_id": biz})
                continue
            futs[biz] = ex.submit(prepare_business, biz, split.businesses[biz], os.path.join(self.work_dir, "biz", biz),
                                  int(cfg.semantic.max_card_tokens))
        split_ctx, rinfo = self.split_context(split)
        await self.wait_llm()
        rs, rep = await build_split_rulespec(rinfo["rb"], rinfo["ref_items"], rinfo["ref_spec"],
                                             llm=self.llm if cfg.rules.use_llm else None,
                                             n_samples=int(cfg.rules.extraction_samples),
                                             min_share=float(cfg.rules.min_vote_share), seed_base=int(cfg.llm.seed_base))
        split_ctx = SplitContext.make(rs, split_ctx.calendar, split_ctx.weekday_names, split_ctx.format_cfg,
                                      split_ctx.rulebook_text, rep, split_ctx.calendar_path)
        split_ctx.save(os.path.join(self.work_dir, "split"))
        self.manifest["rulebook_diff_vs_train"] = rep
        self.events("rulespec", {"changed_sections": rep.get("changed_sections"), "patch": rep.get("patch"),
                                 "uncertain": rep.get("uncertain")})
        memory = ExemplarMemory.load(asset_path(cfg, cfg.memory.path))
        cal_path = asset_path(cfg, cfg.decision.calibrator)
        calibrator = Calibrator.load(cal_path) if cal_path and os.path.exists(cal_path) else Calibrator()
        from dma.solvers.code.pool import SandboxPool
        if cfg.solvers.code.enabled and self.llm.enabled:
            size = min(int(cfg.solvers.code.sandbox_pool), max(1, (os.cpu_count() or 4) - 2), int(cfg.run.max_concurrency))
            self.pool = SandboxPool(size, os.path.join(self.work_dir, "sandbox"), float(cfg.solvers.code.cell_timeout_s),
                                    int(cfg.solvers.code.recycle_after))
        replica_of = assign_replicas(by_biz, self.llm.n_replicas if self.llm else 1)
        excl = {x for x in (a.exclude_memory_businesses or "").split(",") if x} | set(cfg.memory.exclude_businesses or [])
        qqueue: asyncio.Queue = asyncio.Queue()
        build_sem = asyncio.Semaphore(int(cfg.run.max_concurrent_builds))
        loop = asyncio.get_running_loop()

        async def build(biz: str):
            bkp = None
            async with build_sem:
                try:
                    prep = await asyncio.wrap_future(futs[biz])
                    if not prep.get("ok"):
                        self.events("business_prepare_failed", {"business_id": biz, "warnings": prep.get("warnings", [])[:5]})
                        bkp = await asyncio.to_thread(minimal_bkp, prep, split_ctx)
                    else:
                        bsm = BSM.model_validate(prep["proposal"])
                        calls: list = []
                        seed = content_seed(prep["datacard"][:2000], "bsm", base=int(cfg.llm.seed_base))
                        if cfg.semantic.use_llm and self.llm.enabled:
                            import duckdb
                            con = duckdb.connect(prep["ingest"]["db_path"], read_only=True)
                            try:
                                bsm, brep = await refine_bsm(bsm, prep["datacard"], prep.get("readme", ""), prep["profile"],
                                                             con, self.llm, int(cfg.semantic.bsm_samples), seed,
                                                             replica_of.get(biz, 0), calls)
                                notes = await repair_status_maps(bsm, con, self.llm, prep.get("readme", ""), seed + 1,
                                                                 replica_of.get(biz, 0), calls)
                            finally:
                                con.close()
                            self.events("bsm_refined", {"business_id": biz, **brep, "status_notes": notes})
                        bkp = await asyncio.to_thread(finalize_business, prep, bsm, split_ctx)
                except Exception as e:  # noqa: BLE001
                    self.events("business_build_failed", {"business_id": biz, "error": repr(e)[:500],
                                                          "tb": traceback.format_exc()[-1500:]})
            if bkp is not None:
                self.manifest["businesses"].append({
                    "business_id": biz, "files": [f["file"] for f in bkp.profile.get("files", [])],
                    "tables": len(bkp.profile.get("tables", [])),
                    "rows": sum(t.get("n_rows", 0) for t in bkp.profile.get("tables", [])), "build_s": bkp.build_s,
                    "quality": bkp.quality, "packs": bkp.bsm.packs, "gaps": bkp.bsm.gaps, "warnings": bkp.warnings[:10]})
                if cfg.logging.save_bkp:
                    _copy_bkp(bkp, os.path.join(self.log_dir, "bkp", biz))
                self.events("business_ready", {"business_id": biz, "build_s": bkp.build_s, "quality": bkp.quality,
                                               "views": len(bkp.views), "metrics": len(bkp.catalog.metrics)})
            for q in by_biz[biz]:
                await qqueue.put((q, bkp))

        async def worker():
            while True:
                item = await qqueue.get()
                if item is None:
                    qqueue.task_done()
                    return
                q, bkp = item
                if self.clock.past_hard() or self.stop.is_set():
                    qqueue.task_done()          # left for fill_abstain
                    continue
                try:
                    profile, bump = self.current_profile()
                    t0 = time.monotonic()
                    if bkp is None:
                        rec, trace = {"id": q.id, "type": "abstain"}, {"id": q.id, "error": "no knowledge pack for business"}
                    else:
                        rec, trace = await asyncio.wait_for(
                            _answer(q, bkp, split_ctx, self.cfg_for(profile), self.llm, memory, self.pool, calibrator,
                                    replica_of.get(q.business_id, 0), profile, excl, bump),
                            timeout=self.question_timeout())
                    self.durations.append(time.monotonic() - t0)
                except Exception as e:  # noqa: BLE001 - one question never affects another
                    rec = {"id": q.id, "type": "abstain"}
                    trace = {"id": q.id, "business_id": q.business_id, "question": q.question,
                             "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-4000:]}
                    self.stats["errors"] += 1
                try:
                    self.writer.write(rec)
                    trace.setdefault("versions", {"dma": VERSION, "config_sha256": self.manifest["config_sha256"],
                                                  "model": self.cfg.llm.served_name})
                    if cfg.logging.per_question:
                        await asyncio.to_thread(write_log, self.log_dir, q.id, trace)
                    self.stats["done"] += 1
                    t = rec["type"]
                    self.stats["abstain" if t == "abstain" else ("unanswerable" if t == "unanswerable" else "answered")] += 1
                    if self.stats["done"] % 20 == 0:
                        self.events("progress", {**self.stats, "elapsed_s": round(self.clock.elapsed()),
                                                 "remaining": len(todo) - self.stats["done"]})
                finally:
                    qqueue.task_done()

        conc = int(cfg.run.max_concurrency)
        workers = [asyncio.create_task(worker()) for _ in range(conc)]
        await asyncio.gather(*(build(b) for b in futs))
        for b in by_biz:
            if b not in futs:
                for q in by_biz[b]:
                    await qqueue.put((q, None))
        while not qqueue.empty() or qqueue._unfinished_tasks:  # noqa: SLF001
            if self.clock.past_hard() or self.stop.is_set():
                break
            await asyncio.sleep(0.5)
        for _ in workers:
            qqueue.put_nowait(None)
        done, pending = await asyncio.wait(workers, timeout=60)
        for t in pending:
            t.cancel()
        ex.shutdown(wait=False, cancel_futures=True)
        _ = loop

    def current_profile(self) -> tuple[str, float]:
        base = self.cfg.run.profile
        if self.clock.past_soft():
            return "minimal", 0.1
        if self.durations and len(self.durations) >= 20:
            remaining = len(self.all_ids) - self.stats["done"]
            avg = sum(self.durations[-50:]) / len(self.durations[-50:])
            eta = remaining * avg / max(1, int(self.cfg.run.max_concurrency))
            if self.clock.elapsed() + eta > self.clock.soft and base == "accuracy":
                return "fast", 0.0
        return base, 0.0

    def cfg_for(self, profile: str) -> Cfg:
        if profile == self.cfg.run.profile:
            return self.cfg
        if not hasattr(self, "_pcache"):
            self._pcache = {}
        if profile not in self._pcache:
            self._pcache[profile] = apply_profile(self.cfg, profile)
        return self._pcache[profile]

    def question_timeout(self) -> float:
        left = self.clock.hard - self.clock.elapsed()
        return max(30.0, min(float(self.cfg.run.question_timeout_s), left))

    def finish(self) -> None:
        try:
            if self.writer is not None:
                n = self.writer.fill_abstain(self.all_ids)
                if n:
                    self.events("filled_abstain", {"n": n})
                self.writer.close()
        finally:
            self.manifest.update(finished_at=datetime.now().astimezone().isoformat(timespec="seconds"),
                                 wall_s=round(self.clock.elapsed(), 1), n_answered=self.stats["answered"],
                                 n_unanswerable=self.stats["unanswerable"], n_abstain=self.stats["abstain"],
                                 n_errors=self.stats["errors"], llm_stats=self.llm.stats if self.llm else {},
                                 models=_models(self.cfg))
            try:
                with open(os.path.join(self.log_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(self.manifest, f, ensure_ascii=False, indent=1, default=str)
            except OSError:
                pass
            if self.supervisor is not None:
                self.supervisor.stop_all()


async def _answer(q, bkp, split_ctx, cfg, llm, memory, pool, calibrator, replica, profile, excl, bump):
    from dma.pipeline import answer_question
    return await answer_question(q, bkp, split_ctx, cfg, llm, memory, pool, calibrator, replica=replica, profile=profile,
                                 exclude_businesses=excl, threshold_bump=bump)


def _copy_bkp(bkp: BusinessKnowledgePack, dst: str) -> None:
    import shutil
    os.makedirs(dst, exist_ok=True)
    for fn in ("bsm.json", "business_card.md", "datacard.md", "views.sql", "validation.json", "ingest.json"):
        p = os.path.join(bkp.dir, fn)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(dst, fn))


def _env() -> dict:
    env = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("duckdb", "pandas", "vllm", "torch"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "?")
        except Exception:  # noqa: BLE001
            pass
    return env


def _models(cfg: Cfg) -> list[dict]:
    p = "/app/params.json"
    info = {"name": cfg.llm.served_name, "path": cfg.llm.model_path}
    if os.path.exists(p):
        try:
            info.update(json.load(open(p)))
        except (OSError, ValueError):
            pass
    return [info]


def main(argv=None) -> int:
    a = parse_args(argv)
    try:
        cfg = build_cfg(a)
    except Exception as e:  # noqa: BLE001
        print(f"[dma] bad config: {e}", file=sys.stderr)
        return 0
    r = Runner(cfg, a)

    def on_term(signum, frame):  # noqa: ARG001
        r.events("signal", {"signum": signum})
        r.stop.set()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_term)
    try:
        asyncio.run(r.main())
    except BaseException as e:  # noqa: BLE001 - incl. KeyboardInterrupt / SIGTERM
        r.events("fatal", {"error": repr(e)[:500], "tb": traceback.format_exc()[-3000:]})
    finally:
        r.finish()
    r.events("end", {**r.stats, "wall_s": round(r.clock.elapsed(), 1)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
