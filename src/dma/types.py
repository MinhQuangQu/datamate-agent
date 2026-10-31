"""Internal contracts shared across planes (03 §5.6)."""
from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")


class Question(BaseModel):
    id: str
    business_id: str
    question: str
    as_of_date: date


class Candidate(BaseModel):
    solver: str                                   # plan | code | rule (+ "#2" ... for extra samples)
    answer_type: str                              # one of ANSWER_TYPES
    value: Any = None                             # formatter output (JSON-able)
    reason: str | None = None
    month_only: bool = False
    canonical: str = ""                           # f"{type}:{normalized}" per scoring rules
    flags: list[str] = Field(default_factory=list)
    evidence: dict = Field(default_factory=dict)  # plan/sql/steps (for logs & judge)
    tokens: int = 0
    time_used: list[dict] = Field(default_factory=list)   # periods actually used (for time_deterministic)
    repaired: bool = False

    @property
    def base_solver(self) -> str:
        return self.solver.split("#")[0]


class Decision(BaseModel):
    action: Literal["answer", "abstain"]
    chosen: Candidate | None = None
    p_correct: float = 0.0
    threshold: float = 0.5
    source: str = "none"                          # agree_ab | vote | judge | single | none
    features: dict[str, float] = Field(default_factory=dict)
