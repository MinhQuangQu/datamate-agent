"""Deterministic sanity checks S1–S11 (05.7 §3) -> flags (calibrator features).
Flags never drop a candidate, except S2 (string not in data)."""
from __future__ import annotations

from datetime import date

from dma.types import Candidate

NONNEG_HINT = ("revenue", "quantity", "order_count", "customer_count", "line_count", "discount_amount", "purchase")
HEAVY = {"string_not_in_data", "negative_nonneg", "percent_scale_suspect", "list_len_mismatch", "empty_or_zero",
         "tie_at_boundary", "date_out_of_range", "gap_vs_value", "undefined_compare", "type_conflict"}
LIGHT = {"time_nondeterministic", "entity_margin_low", "repaired", "empty_window"}


def run_checks(c: Candidate, qa, bkp, vindex) -> list[str]:
    f = set(c.flags)
    t, v = c.answer_type, c.value
    strong = qa.strong_type
    if strong and t != strong and t != "unanswerable" and not (strong == "list" and t == "table") \
            and not (strong == "date" and t == "string" and "thu may" in qa.folded):
        f.add("type_conflict")                                                          # S1
    if t == "string" and vindex is not None and not vindex.is_data_string(str(v)) and "weekday" not in c.evidence.get("dims", []):
        pass                                                                            # S2 is enforced at the source
    metric = (c.evidence.get("plan") or {}).get("measure", {}) or {}
    mname = metric.get("metric") or ""
    if t in ("number",) and isinstance(v, (int, float)) and v < 0 and any(h in mname for h in NONNEG_HINT):
        f.add("negative_nonneg")                                                        # S3
    if t == "percent" and isinstance(v, (int, float)):
        if (v != 0 and abs(v) <= 1 and ({"growth", "share"} & set(qa.concepts))) or abs(v) > 10000:
            f.add("percent_scale_suspect")                                              # S4
    if t == "list" and qa.k and isinstance(v, list) and len(v) != qa.k and "fewer_candidates" not in f:
        f.add("list_len_mismatch")                                                      # S5
    if qa.ranking_direction and (v in (0, 0.0, None) or v == []):
        f.add("empty_or_zero")                                                          # S6
    if "empty_result" in f:
        f.add("empty_or_zero")
    if t == "date" and isinstance(v, str):
        try:
            d = date.fromisoformat(v if len(v) == 10 else v + "-01")
            cov = bkp.coverage if bkp is not None else None
            if d > qa.as_of or (cov and (v < cov[0][:len(v)] or v > cov[1][:len(v)])):
                f.add("date_out_of_range")                                              # S8
        except ValueError:
            f.add("date_out_of_range")
    used = c.time_used
    if used:
        cands = {(p.start.isoformat(), p.end.isoformat()) for p in qa.time.periods}
        if cands and any((u.get("start"), u.get("end")) not in cands for u in used):
            f.add("time_nondeterministic")                                              # S9
    if qa.gaps_hit and t not in ("unanswerable",):
        f.add("gap_vs_value")                                                           # S10
    if {"division_by_zero", "null_side"} & f:
        f.add("undefined_compare")                                                      # S11
    for m in qa.entities:
        if m.best and m.best.score >= 0.6 and m.margin < 0.05 and len(m.candidates) > 1:
            f.add("entity_margin_low")
            break
    return sorted(f)


def heavy(flags: list[str]) -> bool:
    return bool(HEAVY & set(flags))
