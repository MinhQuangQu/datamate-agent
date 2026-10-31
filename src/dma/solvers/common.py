"""Shared solver plumbing: per-question context, skills (05.6 §6) and the L4 question block."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from dma.llm import prompts as P
from dma.nlu.analyze import weekday_vi

SKILLS = {
    "valid_orders": "Sales metrics count VALID orders only (cancelled orders and orders with returned items are already excluded per the rulebook) — use catalog metrics / dm functions, never re-filter.",
    "best_seller": "\"bán chạy\" without a criterion = rulebook default metric ({best_seller_metric}); a criterion stated in the question wins (doanh thu -> revenue, số đơn -> order_count).",
    "slow_seller": "\"bán chậm/ế\" = ascending rank. {slow_def} Products with zero sales count only if include_zero (master table universe).",
    "profit": "Profit/gross profit = revenue - cost of goods ({cost_method}); use metric gross_profit / cogs. If 'cost' is a gap -> unanswerable.",
    "debt": "Debt/công nợ = outstanding balance up to as_of (metric debt_outstanding / dm.debt). If 'receivables' is a gap -> unanswerable.",
    "stock": "Stock/tồn kho at as_of: metric stock_on_hand / dm.stock. If 'inventory' is a gap -> unanswerable.",
    "expiry": "\"sắp hết hạn\" uses the rulebook vocabulary if defined (dm.expiring); compare with as_of_date.",
    "holiday": "Tết / Trung thu windows come from the rulebook + lunar calendar: use the TIME CANDIDATES / dm.period('Tết năm nay'); never count lunar days yourself.",
    "growth": "Growth % = (current - previous) / previous * 100, percent units, 2 decimals; a decrease is negative. Use op=compare / dm.compare(mode='pct_change').",
    "share": "Share % = part / total * 100 over the same period (op=share / dm.share).",
    "ranking_ties": "Ranking ties -> code ascending. If fewer than k candidates exist, return all of them (do not pad).",
    "answer_types": "Dates 'yyyy-mm-dd'; month questions 'yyyy-mm'; weekday answers use the weekday NAMES of the data/rulebook (e.g. 'Thứ Bảy').",
    "unanswerable": "unanswerable ONLY when the data lacks the concept (gaps). If the concept exists but nothing happened in the period, the answer is 0 / false / an empty-result value, not unanswerable.",
    "cancel": "Questions ABOUT cancelled/returned orders use all orders: metric order_count_all with an order_status filter, or canceled_order_count / returned_order_count / cancel_rate.",
}
CONCEPT_SKILLS = {"best_seller": "best_seller", "slow_seller": "slow_seller", "profit": "profit", "cost": "profit",
                  "debt": "debt", "stock": "stock", "expiry": "expiry", "growth": "growth", "share": "share",
                  "cancel": "cancel", "return": "cancel"}


def skills_block(qa, rs, max_skills: int = 4) -> str:
    names = []
    for c in qa.concepts:
        s = CONCEPT_SKILLS.get(c)
        if s and s not in names:
            names.append(s)
    if any(p.label.startswith(("tet", "mid_autumn")) for p in qa.time.periods):
        names.append("holiday")
    if qa.k or qa.ranking_direction:
        names.append("ranking_ties")
    if qa.gaps_hit or "forecast" in qa.concepts:
        names.append("unanswerable")
    names = names[:max_skills] + ["valid_orders", "answer_types"]
    fmt = {"best_seller_metric": rs.ranking.best_seller_metric,
           "slow_def": f"Rulebook: {rs.ranking.slow_seller_definition}." if rs.ranking.slow_seller_definition else "",
           "cost_method": rs.cost.method}
    return "SKILLS:\n" + "\n".join(f"- {SKILLS[n].format(**fmt)}" for n in dict.fromkeys(names))


def entity_table(qa, k: int = 3) -> str:
    rows = qa.entity_rows(k)
    if not rows:
        return "(none)"
    return "\n".join(f"{r['mention']} | {r['kind']} | {r['code'] or '-'} | {r['name']} | {r['score']:.2f}" for r in rows)


def focus_line(qa, bkp) -> str:
    parts = []
    for m in qa.entities:
        b = m.best
        if b and b.score >= 0.85:
            parts.append(f"{b.kind} {b.code + ' ' if b.code else ''}'{b.name}'")
    cov = bkp.coverage if bkp is not None else None
    if cov:
        parts.append(f"data covers {cov[0]}..{cov[1]}")
    return "; ".join(parts) or "-"


def question_block(qa, bkp, rs, weekday_names: dict[int, str], exemplars: str) -> str:
    return P.QUESTION_BLOCK.format(
        as_of=qa.as_of.isoformat(), as_of_weekday=weekday_vi(qa.as_of, weekday_names), raw=qa.raw,
        expanded=qa.expanded, abbrev_hints="; ".join(qa.abbrev_hints) or "none",
        periods_json=json.dumps(qa.periods_json(), ensure_ascii=False),
        weekdays=list(qa.time.weekdays) if qa.time.weekdays else "none",
        entity_table=entity_table(qa), concepts=", ".join(qa.concepts) or "-",
        gaps_hit=", ".join(qa.gaps_hit) or "-",
        type_cues=", ".join(f"{t}({w:.2f})" for t, w in qa.type_cues[:3]) or "-",
        focus=focus_line(qa, bkp), skills=skills_block(qa, rs), exemplars=exemplars)


@dataclass
class SolverContext:
    cfg: Any
    bkp: Any
    split: Any
    llm: Any
    memory: Any
    pool: Any = None
    replica: int = 0
    exclude_businesses: set = field(default_factory=set)
    calls: list = field(default_factory=list)       # LLM call logs of the current question
    failed: list = field(default_factory=list)      # trajectories that ended without a candidate (for the trace)
    profile: str = "accuracy"

    @property
    def format_cfg(self):
        return self.split.format_cfg
