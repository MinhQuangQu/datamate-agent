"""Arbitration ladder + pairwise judge + calibrated decision (05.7 §2, §4–7)."""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from typing import Awaitable, Callable

from dma.llm import prompts as P
from dma.llm.gateway import content_seed
from dma.nlu.concepts import GAP_REASON_VI
from dma.types import Candidate, Decision
from dma.verify.calibrator import Calibrator
from dma.verify.checks import HEAVY, LIGHT, run_checks

W_MAIN, W_EXTRA, W_RULE = 1.0, 0.8, 0.5


W_RULE_STRONG = 1.5   # deterministic rulebook rule (period outside the data range)
W_VERIFY = 1.5        # verifier agent: checked the disputed point against the data
W_PLAN_ABSENT = 0.5   # plan "unanswerable": the planner sees only the catalog, not the tables, so it cannot know
                      # whether the data lacks something - only that DMQL cannot express it


def weight(c: Candidate) -> float:
    if c.solver == "rule":
        return W_RULE_STRONG if "out_of_coverage" in (c.evidence or {}).get("gaps_hit", []) else W_RULE
    if c.solver == "verify":
        return W_VERIFY
    if c.base_solver == "plan" and c.answer_type == "unanswerable":
        return W_PLAN_ABSENT
    return W_EXTRA if "#" in c.solver else W_MAIN


def rule_candidate(qa) -> Candidate | None:
    if not qa.gaps_hit:
        return None
    g = "out_of_coverage" if "out_of_coverage" in qa.gaps_hit else qa.gaps_hit[0]
    return Candidate(solver="rule", answer_type="unanswerable", canonical="unanswerable",
                     reason=GAP_REASON_VI.get(g, "Dữ liệu không đủ để trả lời"), evidence={"gaps_hit": qa.gaps_hit})


def group(pool: list[Candidate]) -> dict[str, list[Candidate]]:
    g: dict[str, list[Candidate]] = defaultdict(list)
    for c in pool:
        if c.solver == "rule" and c.answer_type != "unanswerable":
            continue
        g[c.canonical].append(c)
    return g


def pick_rep(cands: list[Candidate]) -> Candidate:
    def key(c: Candidate):
        return (c.solver == "rule", c.base_solver != "plan", len(set(c.flags) & HEAVY), len(c.flags), c.solver)
    return sorted(cands, key=key)[0]


def _summ(c: Candidate, max_chars: int = 2500) -> str:
    ev = c.evidence or {}
    if c.base_solver == "plan":
        s = "PLAN: " + json.dumps(ev.get("plan"), ensure_ascii=False) + "\nSQL: " + " ; ".join(ev.get("sql", [])[-2:]) + \
            "\nROWS: " + json.dumps(ev.get("rows_preview", [])[:5], ensure_ascii=False, default=str)
    elif c.base_solver == "code":
        st = ev.get("steps", [])
        s = "\n".join(f"CELL {i + 1}:\n{x.get('code')}\nOUT: {(x.get('observation') or '')[:600]}" for i, x in enumerate(st[-3:]))
    else:
        s = json.dumps(ev, ensure_ascii=False)
    return s[:max_chars]


async def pairwise_judge(qa, ctx, a: Candidate, b: Candidate) -> tuple[Candidate | None, dict]:
    x, y = sorted([a, b], key=lambda c: c.canonical)          # fixed order by canonical: no position bias, deterministic
    card = ctx.bkp.card[:3500]
    msg = P.JUDGE_USER.format(digest=ctx.split.digest_for(qa.raw, str(ctx.cfg.rules.prompt_rulebook)), card=card, raw=qa.raw, as_of=qa.as_of.isoformat(),
                              periods_json=json.dumps(qa.periods_json(), ensure_ascii=False),
                              type_a=x.answer_type, value_a=json.dumps(x.value, ensure_ascii=False), evidence_a=_summ(x),
                              flags_a=x.flags, type_b=y.answer_type, value_b=json.dumps(y.value, ensure_ascii=False),
                              evidence_b=_summ(y), flags_b=y.flags)
    seed = content_seed(qa.raw, qa.as_of, "judge", 0, base=ctx.cfg.llm.seed_base)
    obj, err, _ = await ctx.llm.json("judge", [{"role": "system", "content": P.JUDGE_SYSTEM}, {"role": "user", "content": msg}],
                                     schema=P.JUDGE_SCHEMA, seed=seed, replica=ctx.replica, calls=ctx.calls)
    if not isinstance(obj, dict):
        return None, {"error": err}
    w = obj.get("winner")
    return ({"A": x, "B": y}.get(w), obj)


async def verify_agent(qa, ctx, reps: list[Candidate]) -> Candidate | None:
    """Tool-using referee: a code-solver session that sees the disputed attempts, checks the point where they
    differ against the data and answers itself. Labels are assigned by canonical order (no vote counts shown)."""
    from dma.solvers.code.agent import solve_code
    reps = sorted(reps, key=lambda c: c.canonical)
    blocks = [P.VERIFY_CANDIDATE.format(label=chr(65 + i), atype=c.answer_type,
                                        value=json.dumps(c.value, ensure_ascii=False, default=str)[:600]
                                        if c.answer_type != "unanswerable" else f"unanswerable ({c.reason or ''})"[:400],
                                        evidence=_summ(c, 2000))
              for i, c in enumerate(reps)]
    task = P.VERIFY_TASK.format(candidates="\n".join(blocks))
    seed = content_seed(qa.raw, qa.as_of, "verify", 0, base=ctx.cfg.llm.seed_base)
    return await solve_code(qa, ctx, seed=seed, tag="verify", task=task, stage_prefix="verify")


def _rank(pool: list[Candidate]) -> list[tuple[str, list[Candidate]]]:
    return sorted(group(pool).items(), key=lambda kv: (-sum(weight(c) for c in kv[1]), kv[0]))


async def arbitrate(qa, ctx, first: list[Candidate | None],
                    extra_fns: list[Callable[[], Awaitable[Candidate | None]]]) -> tuple[Candidate | None, dict]:
    arb = ctx.cfg.solvers.arbitration
    pool = [c for c in first if c is not None]
    R = rule_candidate(qa)
    if R:
        pool.append(R)
    for c in pool:
        c.flags = run_checks(c, qa, ctx.bkp, ctx.bkp.vindex) if c.solver != "rule" else c.flags
    info: dict = {"n_first": len([c for c in first if c is not None]), "_pool": pool}
    A = next((c for c in first if c is not None and c.base_solver == "plan"), None)
    B = next((c for c in first if c is not None and c.base_solver == "code"), None)
    if A and B and A.canonical == B.canonical:
        info["source"] = "agree_ab"
        return pick_rep(group(pool)[A.canonical]), info
    if extra_fns:
        extra = await asyncio.gather(*(f() for f in extra_fns), return_exceptions=True)
        errs = [repr(e)[:300] for e in extra if isinstance(e, BaseException)]
        if errs:
            info["extra_errors"] = errs
        for c in extra:
            if isinstance(c, Candidate):
                c.flags = run_checks(c, qa, ctx.bkp, ctx.bkp.vindex)
                pool.append(c)
    ranked = _rank(pool)
    disputed = len(ranked) >= 2 or (len(ranked) == 1 and len(ranked[0][1]) == 1 and ranked[0][1][0].solver != "rule")
    if disputed and arb.get("verify", False) and ctx.pool is not None and ctx.llm is not None and ctx.llm.enabled:
        try:
            V = await verify_agent(qa, ctx, [pick_rep(v) for _, v in ranked[:3]])
        except Exception as e:  # noqa: BLE001 - a failed verifier just falls back to vote / judge
            V, info["verify_error"] = None, repr(e)[:300]
        if V is not None:
            V.flags = run_checks(V, qa, ctx.bkp, ctx.bkp.vindex)
            pool.append(V)
            ranked = _rank(pool)
            info["verify"] = {"canonical": V.canonical[:120],
                              "agrees_with": [c.solver for c in group(pool)[V.canonical] if c is not V]}
    info["groups"] = [{"canonical": k[:120], "weight": round(sum(weight(c) for c in v), 2), "solvers": [c.solver for c in v]}
                      for k, v in ranked]
    if not ranked:
        info["source"] = "none"
        return None, info
    w0 = sum(weight(c) for c in ranked[0][1])
    w1 = sum(weight(c) for c in ranked[1][1]) if len(ranked) > 1 else 0.0
    if any(c.solver == "verify" for c in ranked[0][1]):
        info["source"] = "verify"
        return pick_rep(ranked[0][1]), info
    if len(ranked) == 1 or w0 - w1 >= float(arb.vote_margin):
        info["source"] = "single" if len(pool) == 1 else "vote"
        return pick_rep(ranked[0][1]), info
    if arb.judge and ctx.llm is not None and ctx.llm.enabled:
        a, b = pick_rep(ranked[0][1]), pick_rep(ranked[1][1])
        winner, jinfo = await pairwise_judge(qa, ctx, a, b)
        info["judge"] = jinfo
        info["source"] = "judge"
        if winner is None:
            return None, info
        return winner, info
    info["source"] = "vote_tie"
    return pick_rep(ranked[0][1]), info


def features(chosen: Candidate | None, pool_groups: list[dict], first: list[Candidate | None], qa, ctx, info: dict) -> dict:
    A = next((c for c in first if c is not None and c.base_solver == "plan"), None)
    B = next((c for c in first if c is not None and c.base_solver == "code"), None)
    tot = sum(g["weight"] for g in pool_groups) or 1.0
    win = next((g["weight"] for g in pool_groups if chosen is not None and g["canonical"] == chosen.canonical[:120]), 0.0)
    fl = set(chosen.flags) if chosen else set()
    best_ent = max((m.score for m in qa.entities), default=1.0)
    margin = min((m.margin for m in qa.entities), default=1.0)
    rule_uncertain = int(bool(set(ctx.split.rulespec.uncertain_fields) - {"rulebook_missing"}))
    return {
        "agree_ab": int(bool(A and B and A.canonical == B.canonical)),
        "vote_share": round(win / tot, 4) if pool_groups else (1.0 if chosen else 0.0),
        "n_candidates": sum(len([x for x in g["solvers"] if x != "rule"]) for g in pool_groups)
        if pool_groups else len([c for c in first if c is not None]),
        "plan_ok": int(bool(A and not A.repaired)),
        "code_steps": len((B.evidence or {}).get("steps", [])) if B else 0,
        "exec_errors": int((B.evidence or {}).get("exec_errors", 0)) if B else 0,
        "entity_score": round(best_ent, 4),
        "entity_margin": round(margin, 4),
        "time_deterministic": int("time_nondeterministic" not in fl),
        "empty_or_zero": int("empty_or_zero" in fl),
        "tie_flag": int("tie_at_boundary" in fl),
        "answer_type_prior": 0.0,
        "intent_prior": 0.0,
        "unanswerable_pred": int(bool(chosen and chosen.answer_type == "unanswerable")),
        "self_conf": 0.0,
        "mean_logprob": 0.0,
        "bsm_quality": float(ctx.bkp.quality),
        "rule_uncertain": rule_uncertain,
        "judge_used": int(info.get("source") == "judge"),
        "type_conflict": int("type_conflict" in fl),
        "heavy_flags": len(fl & HEAVY),
        "light_flags": len(fl & LIGHT),
    }


def fallback_prob(feats: dict, source: str) -> float:
    """05.7 §6.3 — used until calibrator.json is trained on out-of-fold data."""
    heavy = feats.get("heavy_flags", 0) > 0
    if feats.get("agree_ab"):
        if not heavy and not feats.get("light_flags"):
            return 0.95
        return 0.80 if not heavy else 0.62
    if source in ("vote", "single") and feats.get("vote_share", 0) >= 2 / 3 and feats.get("n_candidates", 0) >= 2:
        return 0.65 if not heavy else 0.52
    if source == "judge":
        return 0.55 if not heavy else 0.45
    return 0.40


def final_decision(chosen: Candidate | None, feats: dict, cal: Calibrator, source: str, threshold_bump: float = 0.0) -> Decision:
    if chosen is None:
        return Decision(action="abstain", chosen=None, p_correct=0.0, threshold=cal.threshold, source=source, features=feats)
    p = cal.prob(feats) if cal.coef else fallback_prob(feats, source)
    unans = bool(feats.get("unanswerable_pred"))
    thr = (cal.threshold_unanswerable if unans else cal.threshold) + threshold_bump
    return Decision(action="answer" if p > thr else "abstain", chosen=chosen, p_correct=round(p, 4), threshold=round(thr, 4),
                    source=source, features=feats)
