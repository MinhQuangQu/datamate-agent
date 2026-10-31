"""Solver A — Plan Solver (05.5): LLM fills DMQL -> validate/normalize -> deterministic
compile + execute -> Candidate. One repair round with the error; failure -> None."""
from __future__ import annotations

import asyncio
import json
import re
import threading
from datetime import date
from typing import Any

from dma.llm import prompts as P
from dma.llm.gateway import content_seed
from dma.memory.exemplars import render_dmql_examples
from dma.output.formatter import safe_format
from dma.semantic.heuristics import status_map
from dma.solvers.common import SolverContext, question_block
from dma.solvers.plan.compiler import CompileError, Compiler
from dma.solvers.plan.dmql import Plan
from dma.textnorm import fold
from dma.types import Candidate
from dma.verify.canonical import canonicalize

OP_TYPES = {
    "aggregate": {"number", "percent"}, "rank": {"string", "list", "table", "number"},
    "argmax_time": {"date", "string", "number"}, "filter_list": {"list", "table"}, "count": {"number"},
    "exists": {"boolean"}, "compare": {"number", "percent", "boolean"}, "share": {"percent"},
    "lookup": {"number", "string", "date"},
}
STATUS_FIELDS = {"status", "order_status", "trang_thai", "sales_lines.order_status", "sem.sales_lines.order_status",
                 "sem.sales_lines_valid.order_status"}
WEEKDAY_WORDS = {"hai": 1, "ba": 2, "tu": 3, "nam": 4, "sau": 5, "bay": 6, "chu nhat": 7, "cn": 7}
PLAN_SCHEMA = Plan.model_json_schema()


class PlanError(Exception):
    """Validation error with repair context for the LLM."""

    def __init__(self, msg: str, context: str = ""):
        super().__init__(msg)
        self.context = context


def build_messages(qa, ctx: SolverContext, shift: int = 0) -> tuple[list[dict], list]:
    exs = ctx.memory.retrieve(qa.template_key, qa.concepts, qa.strong_type, k=int(ctx.cfg.memory.top_k),
                              max_per_template=int(ctx.cfg.memory.max_per_template),
                              exclude_businesses=ctx.exclude_businesses, shift=shift, need="dmql") if ctx.memory else []
    stable = f"{P.DMQL_DOC}\n\n{ctx.split.digest_for(qa.raw, str(ctx.cfg.rules.prompt_rulebook))}\n\n{ctx.bkp.card}"
    l4 = question_block(qa, ctx.bkp, ctx.split.rulespec, ctx.split.weekday_names, render_dmql_examples(exs))
    msgs = [{"role": "system", "content": P.PLAN_SYSTEM},
            {"role": "user", "content": stable + "\n\n" + l4 + "\n\nReturn the DMQL plan JSON."}]
    return msgs, exs


def _norm_status(v: Any) -> Any:
    if isinstance(v, list):
        return [_norm_status(x) for x in v]
    if isinstance(v, str) and v not in ("completed", "canceled", "returned", "pending", "other"):
        m, _ = status_map([v])
        lab = m[v]
        return lab if lab != "other" else v
    return v


def _weekday(v: Any) -> Any:
    if isinstance(v, list):
        return [_weekday(x) for x in v]
    if isinstance(v, str):
        f = fold(v).replace("thu ", "").strip()
        if f.isdigit():
            n = int(f)
            return n - 1 if 2 <= n <= 8 else n
        return WEEKDAY_WORDS.get(f, v)
    return v


def normalize_plan(plan: Plan, qa, ctx: SolverContext) -> tuple[Plan, list[str]]:
    """Deterministic fix-ups + checks (05.5 §2.2). Raises PlanError for the repair round."""
    notes: list[str] = []
    cat = ctx.bkp.catalog
    vi = ctx.bkp.vindex
    if not plan.answerable:
        return plan, notes
    if plan.op == "custom":
        return plan, notes
    if plan.measure and plan.measure.metric and plan.measure.metric not in cat.metrics:
        raise PlanError(f"unknown metric '{plan.measure.metric}'",
                        f"available metrics: {', '.join(sorted(cat.metrics))}; NOT AVAILABLE (gaps): {', '.join(ctx.bkp.bsm.gaps)}")
    if plan.entity and plan.entity not in cat.entities and plan.entity not in cat.dimensions:
        raise PlanError(f"unknown entity '{plan.entity}'",
                        f"entities: {', '.join(sorted(cat.entities))}; dimensions: {', '.join(sorted(cat.dimensions))}")
    fixed = []
    for f in plan.filters:
        field = f.field
        if field in STATUS_FIELDS or field.endswith(".order_status"):
            field = "sem.sales_lines.order_status"
            v = _norm_status(f.value)
            fixed.append(f.model_copy(update={"field": field, "value": v}))
            continue
        if field == "weekday":
            fixed.append(f.model_copy(update={"value": _weekday(f.value)}))
            continue
        if field in cat.entities:
            vals = f.value if isinstance(f.value, list) else [f.value]
            mapped = []
            for v in vals:
                c = vi.code_for(field, str(v)) if v is not None else None
                if c is None:
                    cands = [(r["code"] or r["name"], r["name"], r["score"]) for r in qa.entity_rows(3) if r["kind"] == field]
                    raise PlanError(f"value {v!r} for entity '{field}' is not a known code/name",
                                    f"ENTITY CANDIDATES for {field}: {cands or 'none'}; use the code")
                if str(c) != str(v):
                    notes.append(f"mapped {field} {v!r} -> {c!r}")
                mapped.append(c)
            fixed.append(f.model_copy(update={"value": mapped if isinstance(f.value, list) else mapped[0]}))
            continue
        if field not in cat.dimensions and "." not in field:
            raise PlanError(f"unknown filter field '{field}'",
                            f"entities: {', '.join(sorted(cat.entities))}; dimensions: {', '.join(sorted(cat.dimensions))}; "
                            "or 'sem.<view>.<column>'")
        fixed.append(f)
    plan = plan.model_copy(update={"filters": fixed})
    if plan.answer_type not in OP_TYPES.get(plan.op, {plan.answer_type}):
        if plan.op == "rank" and plan.answer_type == "date" and plan.entity in ("day", "month"):
            plan = plan.model_copy(update={"op": "argmax_time"})
        else:
            raise PlanError(f"answer_type '{plan.answer_type}' is not compatible with op '{plan.op}'",
                            "op -> answer types: " + json.dumps({k: sorted(v) for k, v in OP_TYPES.items()}))
    if plan.answer_type == "percent" and plan.op == "aggregate" and plan.measure and plan.measure.metric \
            and not plan.measure.metric.endswith(("_pct", "_rate")):
        raise PlanError("percent answers come from op=share, op=compare(pct_change) or *_pct/*_rate metrics",
                        "use op=share for 'chiếm bao nhiêu %' and op=compare with compare_mode=pct_change for growth")
    if plan.op == "compare" and plan.compare_mode == "pct_change" and plan.answer_type != "percent":
        plan = plan.model_copy(update={"answer_type": "percent"})
    if plan.op == "argmax_time" and plan.entity == "month" and plan.answer_type == "date":
        plan = plan.model_copy(update={"month_only": True})
    if plan.k and plan.answer_type == "string" and plan.k > 1:
        plan = plan.model_copy(update={"answer_type": "list"})
    return plan, notes


def _execute(plan: Plan, ctx: SolverContext, as_of: date, timeout_s: float):
    cur = ctx.bkp.cursor()
    timer = threading.Timer(timeout_s, cur.interrupt)
    timer.start()
    try:
        return Compiler(ctx.bkp.catalog, as_of=as_of).run(plan, cur)
    finally:
        timer.cancel()


def result_to_value(plan: Plan, res) -> tuple[str, Any, list[str]]:
    flags = list(res.flags)
    t, v = plan.answer_type, res.value
    if plan.op == "compare":
        t = {"pct_change": "percent", "is_increase": "boolean", "is_decrease": "boolean"}.get(plan.compare_mode or "", t)
        if plan.compare_mode in ("diff", "ratio") and t not in ("number", "percent"):
            t = "number"
    if plan.op == "count":
        t = "number"
    if t == "list" and plan.k and isinstance(v, list) and len(v) < plan.k:
        flags.append("fewer_candidates")
    if isinstance(v, (date,)) and t == "date":
        v = v.isoformat()
    return t, v, flags


async def solve_plan(qa, ctx: SolverContext, seed: int | None = None, shift: int = 0, tag: str = "plan") -> Candidate | None:
    evidence: dict[str, Any] = {}
    cand = await _solve_plan(qa, ctx, seed, shift, tag, evidence)
    if cand is None and ctx.calls is not None:
        # keep the failed plan and its errors in the question trace (otherwise only successes are visible)
        ctx.calls.append({"stage": f"{tag}_failed", "plan": evidence.get("plan"), "attempts": evidence.get("attempts"),
                          "raw": evidence.get("raw")})
    return cand


async def _solve_plan(qa, ctx: SolverContext, seed: int | None, shift: int, tag: str,
                      evidence: dict[str, Any]) -> Candidate | None:
    cfg = ctx.cfg
    seed = seed if seed is not None else content_seed(qa.raw, qa.as_of, "plan", 0, base=cfg.llm.seed_base)
    msgs, exs = build_messages(qa, ctx, shift)
    evidence.update({"exemplars": [e.ex_id for e in exs], "attempts": []})
    plan_obj, err, resp = await ctx.llm.json("plan", msgs, schema=PLAN_SCHEMA, seed=seed, replica=ctx.replica,
                                             validate=Plan.model_validate, calls=ctx.calls)
    evidence["raw"] = resp.text[:1500] if resp else None
    repaired = False
    for attempt in range(int(cfg.solvers.plan.repairs) + 1):
        if plan_obj is not None:
            try:
                plan, notes = normalize_plan(plan_obj, qa, ctx)
                evidence["plan"] = plan.model_dump(mode="json", exclude_none=True)
                evidence["notes"] = notes
                if plan.op == "custom" and plan.answerable:
                    evidence["attempts"].append({"error": "op=custom"})
                    return None
                res = await asyncio.to_thread(_execute, plan, ctx, qa.as_of, float(cfg.solvers.plan.sql_timeout_s))
                evidence["sql"] = res.sql
                evidence["rows_preview"] = [list(map(_jsonable, r)) if isinstance(r, (list, tuple)) else _jsonable(r)
                                            for r in res.rows_preview][:10]
                t, v, flags = result_to_value(plan, res)
                if t == "unanswerable":
                    rec = {"id": "_", "type": "unanswerable", "reason": plan.unanswerable_reason}
                    ferr = None
                else:
                    rec, ferr = safe_format("_", t, v, month_only=plan.month_only, cfg=ctx.format_cfg)
                if ferr:
                    if "empty_result" in flags or v is None:
                        evidence["attempts"].append({"error": f"empty result: {ferr}", "flags": flags})
                        return None          # nothing to rank/return: no candidate rather than a guess
                    raise PlanError(f"result could not be formatted as {t}: {ferr}",
                                    "check answer_type vs op and the k / columns fields")
                times = [x.model_dump(mode="json") for x in (plan.time, plan.compare_time) if x is not None]
                if repaired:
                    flags.append("repaired")
                return Candidate(solver=tag, answer_type=rec["type"], value=rec.get("answer"), reason=rec.get("reason"),
                                 month_only=plan.month_only, canonical=canonicalize(rec, ctx.bkp.vindex), flags=flags,
                                 evidence=evidence, time_used=times, repaired=repaired)
            except PlanError as e:
                err, extra = str(e), e.context
            except CompileError as e:
                err, extra = f"CompileError: {e}", (f"metrics: {', '.join(sorted(ctx.bkp.catalog.metrics))}; "
                                                    f"entities: {', '.join(sorted(ctx.bkp.catalog.entities))}")
            except Exception as e:  # noqa: BLE001 - SQL errors etc.
                err, extra = f"{type(e).__name__}: {str(e)[:600]}", ""
        else:
            extra = ""
        evidence["attempts"].append({"error": err})
        if attempt >= int(cfg.solvers.plan.repairs):
            break
        ctx_txt = extra + f"\nTIME CANDIDATES: {json.dumps(qa.periods_json(), ensure_ascii=False)}"
        rep = msgs + [{"role": "assistant", "content": resp.text if resp else ""},
                      {"role": "user", "content": P.PLAN_REPAIR.format(error_message=err, repair_context=ctx_txt)}]
        plan_obj, err, resp = await ctx.llm.json("plan_repair", rep, schema=PLAN_SCHEMA, seed=seed + 1, replica=ctx.replica,
                                                 validate=Plan.model_validate, calls=ctx.calls)
        evidence["raw"] = resp.text[:1500] if resp else evidence.get("raw")
        repaired = True
    return None


def _jsonable(v: Any) -> Any:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if hasattr(v, "quantize"):
        return float(v)
    return v


_ = re
