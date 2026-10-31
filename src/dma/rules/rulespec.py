"""RuleSpec: the structured, split-specific rulebook (05.3 §2). Defaults = the portal's
examples; the shipped assets/train_rulespec.json (hand-verified on train) overrides them;
changed rulebook items are re-extracted at run time."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from dma.timeparse.resolver import TimeRules


class RevenueRule(BaseModel):
    base: Literal["line_net", "line_gross", "qty_x_price"] = "line_net"
    subtract_order_voucher: bool = False
    include_vat: bool = False
    include_shipping: bool = False
    include_service_fee: bool = True          # "Phí phục vụ ... vẫn tính vào doanh thu"


class ValidOrderRule(BaseModel):
    canceled_statuses: list[str] = ["canceled"]
    exclude_order_if_any_line_returned: bool = True
    exclude_returned_lines_only: bool = False


class RankingRule(BaseModel):
    best_seller_metric: Literal["quantity", "revenue", "order_count"] = "quantity"
    default_order: Literal["desc"] = "desc"
    tie_break: Literal["code_asc", "name_asc"] = "code_asc"
    slow_seller_definition: str | None = None
    include_zero_activity: bool | None = None


class TimeRuleSpec(BaseModel):
    week_start: Literal["monday", "sunday"] = "monday"
    current_period_to_date: bool = True
    last_n_days_includes_today: bool | None = None
    tet_start_lunar: tuple[int, int] = (12, 23)
    tet_end_lunar: tuple[int, int] = (1, 10)
    mid_autumn_lunar: tuple[int, int] = (8, 15)
    mid_autumn_window: tuple[int, int] = (0, 0)
    payday_days: list[int] | None = None
    weekend_isodays: list[int] = [6, 7]
    open_period_clip_to_as_of: bool = True
    bare_month_policy: Literal["latest_not_after_as_of", "current_year"] = "latest_not_after_as_of"
    mtd_compare_same_days: bool = True
    shifts: list[tuple[str, int, int]] = []     # [("ca sáng", 6, 13), ...] start/end HOUR inclusive


class CostRule(BaseModel):
    method: Literal["weighted_avg_by_lot", "fifo", "latest_cost", "static_cost_price"] = "weighted_avg_by_lot"
    lots_cutoff: Literal["all", "until_sale_date", "until_as_of"] | None = None
    # gross profit: cost recorded on the sale line (per sold unit) vs the cost method above
    profit_source: Literal["method", "sale_line"] = "method"


class NumberRule(BaseModel):
    decimals: int = 2
    # money (and an average of money) rounds to whole đồng, half up; percents keep `decimals`
    money_decimals: int = 0
    growth_formula: Literal["(cur-prev)/prev*100"] = "(cur-prev)/prev*100"
    share_formula: Literal["part/total*100"] = "part/total*100"


class VocabEntry(BaseModel):
    term: str
    definition: str
    params: dict = Field(default_factory=dict)


class RuleSpec(BaseModel):
    version: Literal["rulespec/1"] = "rulespec/1"
    revenue: RevenueRule = RevenueRule()
    valid_order: ValidOrderRule = ValidOrderRule()
    ranking: RankingRule = RankingRule()
    time: TimeRuleSpec = TimeRuleSpec()
    cost: CostRule = CostRule()
    numbers: NumberRule = NumberRule()
    vocabulary: list[VocabEntry] = Field(default_factory=list)
    free_text_rules: list[str] = Field(default_factory=list)
    provenance: dict[str, list[str]] = Field(default_factory=dict)
    uncertain_fields: list[str] = Field(default_factory=list)

    # ----------------------------------------------------------- field paths
    def get(self, path: str) -> Any:
        cur: Any = self
        for part in path.split("."):
            cur = cur[part] if isinstance(cur, dict) else getattr(cur, part)
        return cur

    def with_patch(self, patch: dict[str, Any]) -> "RuleSpec":
        d = self.model_dump()
        for path, v in patch.items():
            parts = path.split(".")
            cur = d
            for p in parts[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    raise KeyError(path)
                cur = cur[p]
            if parts[-1] not in cur:
                raise KeyError(path)
            cur[parts[-1]] = v
        return RuleSpec.model_validate(d)


SCALAR_FIELDS = [
    "revenue.base", "revenue.subtract_order_voucher", "revenue.include_vat", "revenue.include_shipping",
    "valid_order.canceled_statuses", "valid_order.exclude_order_if_any_line_returned",
    "valid_order.exclude_returned_lines_only",
    "ranking.best_seller_metric", "ranking.tie_break", "ranking.slow_seller_definition", "ranking.include_zero_activity",
    "time.week_start", "time.current_period_to_date", "time.last_n_days_includes_today", "time.tet_start_lunar",
    "time.tet_end_lunar", "time.mid_autumn_lunar", "time.mid_autumn_window", "time.payday_days",
    "time.weekend_isodays", "time.bare_month_policy", "time.mtd_compare_same_days", "time.shifts",
    "cost.method", "cost.lots_cutoff", "cost.profit_source", "numbers.decimals", "numbers.money_decimals",
]


def field_schema(path: str) -> dict:
    sec, name = path.split(".")
    model = RuleSpec.model_fields[sec].annotation
    sch = model.model_json_schema()["properties"][name]                # type: ignore[union-attr]
    return {k: v for k, v in sch.items() if k in ("type", "enum", "anyOf", "items", "prefixItems", "default")}


def time_rules_from(spec: TimeRuleSpec) -> TimeRules:
    return TimeRules(week_start_isoday=1 if spec.week_start == "monday" else 7,
                     current_period_to_date=spec.current_period_to_date,
                     last_n_includes_today=True if spec.last_n_days_includes_today is None else spec.last_n_days_includes_today,
                     tet_start_lunar=tuple(spec.tet_start_lunar), tet_end_lunar=tuple(spec.tet_end_lunar),
                     mid_autumn_lunar=tuple(spec.mid_autumn_lunar), mid_autumn_window=tuple(spec.mid_autumn_window),
                     weekend_isodays=tuple(spec.weekend_isodays), bare_month_policy=spec.bare_month_policy,
                     mtd_compare_same_days=spec.mtd_compare_same_days,
                     shifts=tuple((n, int(a), int(b)) for n, a, b in spec.shifts))


_WD_VI = {1: "Thứ Hai", 7: "Chủ nhật"}


def render_digest(rs: RuleSpec) -> str:
    """Prompt layer L2 (09 §1.3)."""
    r = rs.revenue
    base = {"line_net": "sum of line amounts AFTER line-level discount",
            "line_gross": "sum of line amounts BEFORE discount", "qty_x_price": "quantity × unit price"}[r.base]
    rev = f"{base}; order voucher {'subtracted' if r.subtract_order_voucher else 'NOT subtracted'}; " \
          f"VAT {'included' if r.include_vat else 'excluded'}; shipping {'included' if r.include_shipping else 'excluded'}."
    v = rs.valid_order
    if v.exclude_returned_lines_only:
        vo = "exclude cancelled orders; returned LINES are excluded (the rest of the order counts)."
    elif v.exclude_order_if_any_line_returned:
        vo = "exclude cancelled orders AND whole orders having any returned item."
    else:
        vo = "exclude cancelled orders only."
    t = rs.time
    ws = "Thứ Hai" if t.week_start == "monday" else "Chủ nhật"
    we = "Chủ nhật" if t.week_start == "monday" else "Thứ Bảy"
    mw = "" if tuple(t.mid_autumn_window) == (0, 0) else f" (−{t.mid_autumn_window[0]}/+{t.mid_autumn_window[1]} days)"
    lastn = {None: "unknown (default: includes today)", True: "includes today", False: "excludes today"}[t.last_n_days_includes_today]
    cost = {"weighted_avg_by_lot": "weighted average purchase price over lots", "fifo": "FIFO by lot",
            "latest_cost": "latest purchase price", "static_cost_price": "cost price column of the product master"}[rs.cost.method]
    if rs.cost.lots_cutoff:
        cost += f" (lots: {rs.cost.lots_cutoff})"
    if rs.cost.profit_source == "sale_line":
        cost += "; gross profit uses the cost written on each sale line x quantity in the sold unit"
    vocab = "; ".join(f'"{e.term}" = {e.definition}' for e in rs.vocabulary) or "none"
    lines = [
        "## RULEBOOK DIGEST (authoritative for this split)",
        f"- Revenue: {rev}",
        f"- Valid orders: {vo}",
        f"- Best-selling (\"bán chạy\"): by {rs.ranking.best_seller_metric}; ranking descending; ties -> "
        f"{'code' if rs.ranking.tie_break == 'code_asc' else 'name'} ascending; valid orders only.",
        f"- Week = {ws} .. {we}. \"this week/month/quarter/year\" = start of period .. as_of_date (inclusive)"
        if t.current_period_to_date else f"- Week = {ws} .. {we}. \"this week/month/...\" = the whole calendar period.",
        f"- \"N days ago/last N days\": {lastn}.",
        f"- Tết window: lunar {t.tet_start_lunar[1]}/{t.tet_start_lunar[0]} of the previous year .. lunar "
        f"{t.tet_end_lunar[1]}/{t.tet_end_lunar[0]}. Mid-autumn: lunar {t.mid_autumn_lunar[1]}/{t.mid_autumn_lunar[0]}{mw}.",
        f"- Cost of goods: {cost}.",
        f"- Numbers: {rs.numbers.decimals} decimals; money amounts (incl. averages of money) rounded half up to "
        f"{rs.numbers.money_decimals} decimals; growth % = (current - previous) / previous * 100; share % = part / total * 100.",
        f"- Vocabulary: {vocab}",
    ]
    if rs.ranking.slow_seller_definition:
        lines.append(f"- Slow seller (\"bán chậm\"): {rs.ranking.slow_seller_definition}")
    if rs.free_text_rules and len(rs.free_text_rules) <= 25:
        lines.append("- Other rules (verbatim):")
        lines += [f"  * {x}" for x in rs.free_text_rules]
    if rs.uncertain_fields:
        lines.append(f"- (uncertain, read the verbatim rules: {', '.join(rs.uncertain_fields)})")
    return "\n".join(lines)
