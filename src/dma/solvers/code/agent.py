"""Solver B — Code Solver: CodeAct loop in the sandbox with the `dm` toolkit (05.6 §2–3, §7, §9)."""
from __future__ import annotations

import os
import re
from typing import Any

from dma.llm import prompts as P
from dma.llm.gateway import LLMUnavailable, content_seed
from dma.memory.exemplars import render_code_examples
from dma.output.formatter import safe_format
from dma.solvers.common import SolverContext, question_block
from dma.types import Candidate
from dma.verify.canonical import canonicalize

_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_first_python_block(text: str) -> tuple[str | None, int]:
    blocks = _BLOCK.findall(text or "")
    if blocks:
        return blocks[0].strip(), len(blocks)
    m = re.search(r"```(?:python|py)?\s*\n(.*)$", text or "", re.S)       # unterminated block (max_tokens)
    if m and m.group(1).strip():
        return m.group(1).strip(), 1
    return None, 0


def build_messages(qa, ctx: SolverContext, max_steps: int, shift: int = 0, task: str = "") -> tuple[list[dict], list]:
    exs = ctx.memory.retrieve(qa.template_key, qa.concepts, qa.strong_type, k=int(ctx.cfg.memory.top_k),
                              max_per_template=int(ctx.cfg.memory.max_per_template),
                              exclude_businesses=ctx.exclude_businesses, shift=shift, need="code") if ctx.memory else []
    cat = ctx.bkp.catalog
    l1 = P.TOOLKIT_DOC.replace("{metric_names}", ", ".join(sorted(cat.metrics)) or "(none)") \
        .replace("{entity_names}", ", ".join(sorted(cat.entities)) or "(none)")
    stable = f"{l1}\n\n{ctx.split.digest_for(qa.raw, str(ctx.cfg.rules.prompt_rulebook))}\n\n{ctx.bkp.card}"
    l4 = question_block(qa, ctx.bkp, ctx.split.rulespec, ctx.split.weekday_names, render_code_examples(exs))
    msgs = [{"role": "system", "content": P.CODE_SYSTEM.replace("{max_steps}", str(max_steps))},
            {"role": "user", "content": stable + "\n\n" + l4 + ("\n\n" + task if task else "")}]
    return msgs, exs


def hint_for(obs: str, ok: bool, step: int, max_steps: int, qa, ctx: SolverContext) -> str:
    hints = []
    if step == max_steps - 2:
        hints.append(P.LAST_STEP_HINT)
    low = obs.lower()
    if ok and (re.search(r"empty dataframe|\[value\] 0(\.0)?$|\[value\] none$|columns: \[code, name, value\]\s*index: \[\]", low)
               or "empty_result" in low or "empty_window" in low):
        cov = ctx.bkp.coverage
        hints.append(P.EMPTY_HINT.format(period=[p.as_json() for p in qa.time.periods][:2],
                                         coverage=f"{cov[0]}..{cov[1]}" if cov else "?"))
    if not ok and ("keyerror" in low or "binder error" in low or "catalog error" in low):
        hints.append("Check names with dm.tables() / dm.schema('<table>') before querying.")
    return ("\n" + "\n".join(hints)) if hints else ""


async def solve_code(qa, ctx: SolverContext, seed: int | None = None, shift: int = 0, tag: str = "code",
                     task: str = "", stage_prefix: str = "code") -> Candidate | None:
    """`task` (appended after the shared prompt prefix) turns the same loop into another role, e.g. the verifier."""
    cfg = ctx.cfg
    cc = cfg.solvers.code
    max_steps = int(cc.max_steps)
    seed = seed if seed is not None else content_seed(qa.raw, qa.as_of, "code", 0, base=cfg.llm.seed_base)
    msgs, exs = build_messages(qa, ctx, max_steps, shift, task)
    steps: list[dict[str, Any]] = []
    errors, tokens = 0, 0
    budget = int(cc.max_tokens_per_question)
    split_dir = ctx.split.dir
    async with ctx.pool.acquire(ctx.bkp.business_id, ctx.bkp.dir, split_dir) as kernel:
        import asyncio
        r0 = await asyncio.to_thread(kernel.run, f"dm._bind_question({qa.as_of.isoformat()!r}, {qa.raw!r})", 30)
        if not r0.ok:
            raise RuntimeError(f"toolkit bind failed: {r0.error}")
        for step in range(max_steps):
            first = step == 0
            stage = f"{stage_prefix}_first" if first else f"{stage_prefix}_step"
            if tokens >= budget:
                break
            try:
                out = await ctx.llm.chat(stage, msgs, seed=seed + step, replica=ctx.replica, calls=ctx.calls)
            except LLMUnavailable:
                break
            tokens += out.completion_tokens
            code, nblocks = extract_first_python_block(out.text)
            if code is None:
                msgs = msgs + [{"role": "assistant", "content": out.text or "(empty)"},
                               {"role": "user", "content": P.PROTOCOL_REMINDER}]
                steps.append({"code": None, "observation": "no code block"})
                continue
            res = await asyncio.to_thread(kernel.run, code, float(cc.cell_timeout_s))
            obs = res.observation()
            if len(obs) > int(cfg.logging.max_observation_chars):
                obs = obs[: int(cfg.logging.max_observation_chars)] + "\n...[truncated]"
            steps.append({"code": code, "observation": obs, "ok": res.ok})
            if res.final is not None:
                return _to_candidate(res.final, steps, ctx, errors, tokens, tag, exs)
            errors += int(not res.ok)
            note = "" if nblocks <= 1 else "\n(only the first python block was executed)"
            msgs = msgs + [{"role": "assistant", "content": out.text},
                           {"role": "user", "content": "[observation]\n" + obs + note + hint_for(obs, res.ok, step, max_steps, qa, ctx)}]
    ctx.failed.append({"solver": tag, "steps": [{"code": (x.get("code") or "")[:1500], "observation": (x.get("observation") or "")[:800]}
                                                for x in steps]})
    return None


def _to_candidate(final: dict, steps: list, ctx: SolverContext, errors: int, tokens: int, tag: str, exs) -> Candidate | None:
    t = final.get("type")
    rec, err = safe_format("_", t, final.get("value"), reason=final.get("reason"),
                           month_only=bool(final.get("month_only", False)), cfg=ctx.format_cfg)
    if err or rec["type"] == "abstain":
        steps.append({"code": None, "observation": f"final_answer rejected by formatter: {err}"})
        return None
    flags = list(final.get("warnings") or [])
    times = []
    for c in final.get("dm_calls") or []:
        flags += [f for f in (c.get("flags") or []) if f in ("tie_at_boundary", "empty_result", "empty_window",
                                                              "division_by_zero", "null_side")]
        plan = c.get("plan") or {}
        for k in ("time", "compare_time"):
            if plan.get(k):
                times.append({"start": plan[k]["start"], "end": plan[k]["end"]})
    if errors:
        flags.append("exec_errors")
    return Candidate(solver=tag, answer_type=rec["type"], value=rec.get("answer"), reason=rec.get("reason"),
                     month_only=bool(final.get("month_only", False)), canonical=canonicalize(rec, ctx.bkp.vindex),
                     flags=sorted(set(flags)), tokens=tokens, time_used=times,
                     evidence={"steps": steps, "exemplars": [e.ex_id for e in exs], "exec_errors": errors,
                               "dm_calls": final.get("dm_calls", [])[-6:]})


_ = os
