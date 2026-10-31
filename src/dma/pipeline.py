"""answer_question(): one question end to end (04 §4.2). Never raises for solver errors:
a failed solver is just a missing candidate; the decision layer abstains when unsure."""
from __future__ import annotations

import asyncio
import time
import traceback
from typing import Any

from dma.llm.gateway import content_seed
from dma.nlu.analyze import analyze_question
from dma.output.formatter import safe_format
from dma.solvers.code.agent import solve_code
from dma.solvers.common import SolverContext
from dma.solvers.plan.solver import solve_plan
from dma.types import Candidate, Question
from dma.verify.arbiter import arbitrate, features, final_decision
from dma.verify.checks import HEAVY


async def _safe(name: str, coro, errors: list, timing: dict) -> Candidate | None:
    t0 = time.monotonic()
    try:
        return await coro
    except Exception as e:  # noqa: BLE001
        errors.append({"stage": name, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]})
        return None
    finally:
        timing[name] = timing.get(name, 0) + int(1000 * (time.monotonic() - t0))


async def answer_question(q: Question, bkp, split, cfg, llm, memory, pool, calibrator, replica: int = 0,
                          profile: str = "accuracy", exclude_businesses: set | None = None,
                          threshold_bump: float = 0.0) -> tuple[dict, dict]:
    t0 = time.monotonic()
    timing: dict[str, int] = {}
    errors: list[dict] = []
    ctx = SolverContext(cfg=cfg, bkp=bkp, split=split, llm=llm, memory=memory, pool=pool, replica=replica,
                        exclude_businesses=set(exclude_businesses or ()), calls=[], profile=profile)
    tn = time.monotonic()
    if not hasattr(bkp.vindex, "_extra_words"):
        bkp.attach_lexicon(split.rulebook_text)
    qa = analyze_question(q.question, q.as_of_date, bkp.vindex, bkp.bsm.gaps, split.time_rules, split.calendar, bkp.coverage)
    timing["nlu"] = int(1000 * (time.monotonic() - tn))
    use_llm = bool(llm is not None and llm.enabled)
    plan_on = use_llm and bool(cfg.solvers.plan.enabled) and bool(bkp.catalog.metrics or bkp.catalog.generic_sources)
    code_on = use_llm and bool(cfg.solvers.code.enabled) and pool is not None
    A = B = None
    if profile == "minimal":
        if plan_on:
            A = await _safe("plan", solve_plan(qa, ctx), errors, timing)
        if A is None and code_on:
            B = await _safe("code", solve_code(qa, ctx), errors, timing)
    else:
        tasks = []
        if plan_on:
            tasks.append(_safe("plan", solve_plan(qa, ctx), errors, timing))
        if code_on:
            tasks.append(_safe("code", solve_code(qa, ctx), errors, timing))
        res = await asyncio.gather(*tasks)
        it = iter(res)
        A = next(it) if plan_on else None
        B = next(it) if code_on else None
    arb = cfg.solvers.arbitration
    n_code, n_plan = int(arb.extra_code_samples), int(arb.extra_plan_samples)
    if profile == "minimal":
        n_code = n_plan = 0
    if profile == "fast" and A is not None and not (set(A.flags) & HEAVY):
        n_code, n_plan = min(n_code, 1), 0
    base = cfg.llm.seed_base
    extra_fns = []
    if not (A and B and A.canonical == B.canonical) and use_llm:
        for i in range(n_code if code_on else 0):
            extra_fns.append(lambda i=i: _safe(f"code#{i + 2}", solve_code(qa, ctx, seed=content_seed(qa.raw, qa.as_of, "code", i + 2, base=base),
                                                                           shift=i + 1, tag=f"code#{i + 2}"), errors, timing))
        for i in range(n_plan if plan_on else 0):
            extra_fns.append(lambda i=i: _safe(f"plan#{i + 2}", solve_plan(qa, ctx, seed=content_seed(qa.raw, qa.as_of, "plan", i + 2, base=base),
                                                                           shift=i + 1, tag=f"plan#{i + 2}"), errors, timing))
    tv = time.monotonic()
    chosen, info = await arbitrate(qa, ctx, [A, B], extra_fns)
    pool_all = info.pop("_pool", [])
    feats = features(chosen, info.get("groups", []), [A, B], qa, ctx, info)
    source = info.get("source", "none")
    dec = final_decision(chosen, feats, calibrator, source, threshold_bump)
    timing["verify"] = int(1000 * (time.monotonic() - tv))
    rec: dict[str, Any] = {"id": q.id, "type": "abstain"}
    if dec.action == "answer" and dec.chosen is not None:
        c = dec.chosen
        if c.answer_type == "unanswerable":
            rec = {"id": q.id, "type": "unanswerable", **({"reason": c.reason[:300]} if c.reason else {})}
        else:
            rec, ferr = safe_format(q.id, c.answer_type, c.value, month_only=c.month_only, cfg=split.format_cfg)
            if ferr:
                errors.append({"stage": "format", "error": ferr})
    timing["total"] = int(1000 * (time.monotonic() - t0))
    trace = {
        "id": q.id, "business_id": q.business_id, "question": q.question, "as_of_date": q.as_of_date.isoformat(),
        "final": {k: v for k, v in rec.items() if k != "id"},
        "decision": {"action": dec.action, "p_correct": dec.p_correct, "threshold": dec.threshold, "source": source,
                     "chosen_solver": dec.chosen.solver if dec.chosen else None},
        "timing_ms": timing,
        "nlu": qa.as_json(),
        "candidates": [_cand_log(c) for c in pool_all],
        "arbitration": {k: v for k, v in info.items() if k != "source"},
        "features": feats,
        "llm_calls": ctx.calls,
        "failed_trajectories": ctx.failed,
        "errors": errors,
        "profile": profile,
    }
    return rec, trace


def _cand_log(c: Candidate) -> dict:
    return c.model_dump(mode="json")
