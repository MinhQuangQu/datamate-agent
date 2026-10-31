"""DMQL v1 — the DataMate Query Language emitted by the Plan Solver (Solver A).

The LLM fills a small, typed JSON plan; a deterministic compiler turns it into SQL
over the semantic views. Same idea as ThoughtSpot search tokens / dbt MetricFlow
queries / LookML field picks: the model chooses WHAT, code decides HOW.
`Plan.model_json_schema()` is passed to vLLM structured outputs.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AnswerType = Literal["number", "percent", "string", "date", "boolean", "list", "table", "unanswerable"]
Op = Literal["aggregate", "rank", "compare", "share", "argmax_time", "exists", "count", "lookup",
             "filter_list", "custom"]
FilterOp = Literal["=", "!=", "in", "not_in", ">", ">=", "<", "<=", "between", "contains"]


class Filter(BaseModel):
    field: str = Field(description="entity (product, customer, category, branch, staff, supplier...), "
                                   "dimension (weekday, month, hour, day), entity attribute ('product.active_ingredient', "
                                   "'product.is_prescription', ...; text compared ignoring case/accents) or 'table.column'")
    op: FilterOp = "="
    value: Any = Field(description="entity CODES after linking (e.g. 'SP0042'), numbers, or ISO dates")


class TimeRange(BaseModel):
    start: date
    end: date
    expr: str | None = Field(None, description="surface text, e.g. 'tuần rồi'")

    @model_validator(mode="after")
    def _order(self) -> "TimeRange":
        if self.end < self.start:
            raise ValueError("time.end < time.start")
        return self


class Measure(BaseModel):
    metric: str | None = Field(None, description="rule-aware catalog metric, e.g. revenue, quantity, order_count")
    column: str | None = Field(None, description="generic 'table.column' when no catalog metric fits")
    agg: Literal["sum", "count", "count_distinct", "avg", "min", "max"] | None = None

    @model_validator(mode="after")
    def _one(self) -> "Measure":
        if not self.metric and not (self.column and self.agg):
            raise ValueError("measure needs `metric` or (`column` + `agg`)")
        return self


class Plan(BaseModel):
    version: Literal["dmql/1"] = "dmql/1"
    answerable: bool = True
    unanswerable_reason: str | None = None
    answer_type: AnswerType
    op: Op
    measure: Measure | None = None
    entity: str | None = Field(None, description="group-by entity or time dimension for rank/count/filter_list/argmax_time")
    filters: list[Filter] = Field(default_factory=list)
    time: TimeRange | None = None
    compare_time: TimeRange | None = None
    balance_date: date | None = Field(None, description="cutoff day of as-of balances (debt_outstanding, debtor_count, "
                                                        "stock_on_hand): 'tính đến hết ngày X'; default = today")
    compare_mode: Literal["diff", "pct_change", "ratio", "is_increase", "is_decrease"] | None = None
    having: Filter | None = Field(None, description="condition on the measure value, field must be 'measure'")
    include_zero: bool = Field(False, description="rank/filter over ALL entities incl. those with no activity")
    order: Literal["desc", "asc"] = "desc"
    k: int | None = Field(None, ge=1, le=100)
    lookup_attribute: str | None = None
    share_drop_fields: list[str] | None = Field(None, description="filters removed for the share denominator")
    columns: list[str] | None = Field(None, description="output column names for table answers")
    month_only: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def _consistency(self) -> "Plan":
        if not self.answerable:
            self.answer_type = "unanswerable"
            return self
        if self.answer_type == "unanswerable":
            self.answerable = False
            return self
        needs_measure = {"aggregate", "rank", "compare", "share", "argmax_time", "filter_list"}
        if self.op in needs_measure and self.measure is None:
            raise ValueError(f"op={self.op} requires `measure`")
        if self.op in {"rank", "argmax_time", "filter_list"} and not self.entity:
            raise ValueError(f"op={self.op} requires `entity`")
        if self.op == "compare" and (self.time is None or self.compare_time is None or self.compare_mode is None):
            raise ValueError("compare requires time, compare_time and compare_mode")
        if self.op == "rank" and self.k is None:
            self.k = 1
        return self
