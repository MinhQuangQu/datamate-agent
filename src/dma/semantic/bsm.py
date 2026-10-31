"""Business Semantic Model (05.2 §3) + ontology loader."""
from __future__ import annotations

import functools
import os
from typing import Literal

import yaml
from pydantic import BaseModel, Field

PACK_ENTITIES = ("sales_lines", "orders", "products", "customers", "purchase_lots", "inventory", "receivables",
                 "services", "appointments", "staff", "branches", "suppliers", "expenses")
STATUS_LABELS = ("completed", "canceled", "returned", "pending", "other")
GAP_CONCEPTS = ("cost", "staff", "branch", "expiry", "time_of_day", "channel", "payment_method", "receivables",
                "inventory", "voucher", "vat", "shipping", "supplier", "customer", "category", "revenue")


class ColumnRole(BaseModel):
    table: str
    column: str
    role: str
    confidence: float = 1.0
    unit_multiplier: float = 1.0
    value_map: dict[str, str] | None = None
    note: str | None = None


class LogicalTable(BaseModel):
    name: str
    description: str = ""
    grain: str = "other"
    primary_key: list[str] = Field(default_factory=list)
    pack_entity: str | None = None
    date_col: str | None = None


class Relationship(BaseModel):
    from_table: str
    from_column: str
    to_table: str
    to_column: str
    kind: Literal["many_to_one", "one_to_one"] = "many_to_one"
    coverage: float = 1.0


class BSM(BaseModel):
    version: Literal["bsm/1"] = "bsm/1"
    domain_guess: str = "other"
    packs: list[str] = Field(default_factory=list)
    currency: str = "VND"
    tables: list[LogicalTable] = Field(default_factory=list)
    roles: list[ColumnRole] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    date_coverage: dict[str, list[str]] = Field(default_factory=dict)
    gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    readme_facts: list[str] = Field(default_factory=list)
    interpretation: dict[str, str] = Field(default_factory=dict)   # e.g. {"line_net": "q*p-d", "status_map": "llm"}
    quality: float = 1.0
    unit_conversions: list[list] = Field(default_factory=list)       # [product, big_unit, factor, base_unit] from README
    data_end: str | None = None                                        # "Dữ liệu có đến hết dd/mm/yyyy"
    data_start: str | None = None
    readme_roles: dict[str, str] = Field(default_factory=dict)         # "table.column" -> role stated in README

    # ---------------------------------------------------------------- lookup
    def table_for(self, pack_entity: str) -> str | None:
        ts = self.tables_for(pack_entity)
        return ts[0] if ts else None

    def tables_for(self, pack_entity: str) -> list[str]:
        return [t.name for t in self.tables if t.pack_entity == pack_entity]

    def roles_of(self, table: str) -> dict[str, ColumnRole]:
        out: dict[str, ColumnRole] = {}
        for r in self.roles:
            if r.table == table and r.role not in out:
                out[r.role] = r
        return out

    def col(self, table: str | None, role: str) -> str | None:
        if table is None:
            return None
        r = self.roles_of(table).get(role)
        return r.column if r else None

    def has_role(self, role: str) -> bool:
        return any(r.role == role for r in self.roles)

    def has(self, pack_entity: str) -> bool:
        return any(t.pack_entity == pack_entity for t in self.tables)

    def table(self, name: str) -> LogicalTable | None:
        return next((t for t in self.tables if t.name == name), None)


ROLE_COMPAT = {
    "id": {"id", "string", "number"},
    "string": {"string", "id"},
    "number": {"number"},
    "percent": {"percent", "number"},
    "date": {"date", "datetime"},
    "datetime": {"datetime", "date"},
    "bool": {"bool", "string", "number"},
}


def dtype_compatible(role_dtype: str, kind: str) -> bool:
    return kind in ROLE_COMPAT.get(role_dtype, {kind})


def _default_ontology_path() -> str:
    env = os.environ.get("DMA_ONTOLOGY")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "..", "..", "..", "assets", "ontology.yaml"), "/app/assets/ontology.yaml"):
        if os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("assets/ontology.yaml not found (set DMA_ONTOLOGY)")


@functools.lru_cache(maxsize=4)
def load_ontology(path: str | None = None) -> dict:
    with open(path or _default_ontology_path(), encoding="utf-8") as f:
        ont = yaml.safe_load(f)
    def s(x) -> str:   # YAML 1.1 turns bare `no`/`yes` into booleans
        return ("yes" if x else "no") if isinstance(x, bool) else str(x).strip()
    for r in ont["roles"].values():
        r["syn"] = [s(x) for x in r.get("syn", [])]
    ont["table_hints"] = {k: [s(x) for x in v] for k, v in ont.get("table_hints", {}).items()}
    return ont
