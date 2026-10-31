"""DMQL -> DuckDB SQL compiler + executor (deterministic half of the Plan Solver).

The Catalog is built per business from the Business Semantic Model (BSM) and the
split RuleSpec: metric expressions already encode the rulebook (valid orders,
line-level discount, no VAT/shipping...). The compiler only composes them, so a rule
is implemented once and reused by every question.

Tie-break for rankings (rulebook default): metric DESC, then entity code ASC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from dma.solvers.plan.dmql import Filter, Plan, TimeRange


class CompileError(Exception):
    """Plan cannot be compiled with this catalog -> route to the Code Solver."""


@dataclass
class EntityDef:
    name: str
    key: str                       # key column inside fact sources (the object CODE)
    label: str                     # display-name column inside fact sources
    dim_table: str | None = None   # master table (universe for include_zero / lookup)
    dim_key: str | None = None
    dim_label: str | None = None
    attributes: dict[str, str] = field(default_factory=dict)   # lookup attr -> column in dim_table


@dataclass
class DimensionDef:
    name: str
    expr: str                      # SQL over fact alias s, may use {date}
    kind: str                      # date | month | weekday | int | text


@dataclass
class MetricDef:
    name: str
    source: str                    # relation (view) name
    expr: str                      # aggregate SQL over alias s
    date_col: str
    entities: tuple[str, ...]      # entities available in this source
    additive: bool = True          # SUM/COUNT-like -> empty window means 0
    money_decimals: int | None = None   # money metric: scalar rounded half up (RuleSpec numbers.money_decimals)
    description: str = ""


@dataclass
class Catalog:
    metrics: dict[str, MetricDef]
    entities: dict[str, EntityDef]
    dimensions: dict[str, DimensionDef] = field(default_factory=dict)
    generic_sources: dict[str, str] = field(default_factory=dict)   # table -> date column (or "")
    weekday_names: dict[int, str] = field(default_factory=lambda: {
        1: "Thứ Hai", 2: "Thứ Ba", 3: "Thứ Tư", 4: "Thứ Năm", 5: "Thứ Sáu", 6: "Thứ Bảy", 7: "Chủ nhật"})


@dataclass
class PlanResult:
    answer_type: str
    value: Any
    sql: list[str] = field(default_factory=list)
    rows: list[Any] = field(default_factory=list)          # full result (bounded by k / LIMIT)
    rows_preview: list[Any] = field(default_factory=list)  # first 10, for logs
    flags: list[str] = field(default_factory=list)


def default_dimensions(date_col: str = "{date}", ts_col: str | None = None) -> dict[str, DimensionDef]:
    """`{date}` is replaced by the date column of the metric being compiled."""
    d = {
        "day": DimensionDef("day", f"CAST(s.{date_col} AS DATE)", "date"),
        "month": DimensionDef("month", f"strftime(s.{date_col}, '%Y-%m')", "month"),
        "weekday": DimensionDef("weekday", f"isodow(s.{date_col})", "weekday"),
        "year": DimensionDef("year", f"year(s.{date_col})", "int"),
        "quarter": DimensionDef("quarter", f"quarter(s.{date_col})", "int"),
        "day_of_month": DimensionDef("day_of_month", f"day(s.{date_col})", "int"),
    }
    if ts_col:
        d["hour"] = DimensionDef("hour", f"hour(s.{ts_col})", "int")
    return d


def round_money(v: Any, decimals: int | None) -> Any:
    if decimals is None or v is None or isinstance(v, bool):
        return v
    q = Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
    return int(q) if decimals <= 0 else float(q)


class Compiler:
    def __init__(self, catalog: Catalog, as_of: date | None = None):
        self.cat = catalog
        self.as_of = as_of           # replaces `$as_of` in metric sources (balances "đến nay")
        self._date = ""              # date column of the metric being compiled (for `{date}` dimensions)

    def _dim(self, dd: DimensionDef) -> str:
        return dd.expr.replace("{date}", self._date or "order_date")

    # ------------------------------------------------------------ helpers
    def _measure(self, plan: Plan) -> tuple[str, str, str, MetricDef | None]:
        """-> (source, aggregate_expr, date_col, metricdef)"""
        m = plan.measure
        if m is None:
            raise CompileError("missing measure")
        if m.metric:
            md = self.cat.metrics.get(m.metric)
            if md is None:
                raise CompileError(f"unknown metric '{m.metric}'")
            src = md.source
            if "$as_of" in src:
                cut = plan.balance_date or self.as_of or date(9999, 12, 31)
                src = src.replace("$as_of", f"DATE '{cut.isoformat()}'")
            self._date = md.date_col
            return src, md.expr, md.date_col, md
        table, _, col = (m.column or "").rpartition(".")
        if table not in self.cat.generic_sources:
            raise CompileError(f"unknown table for generic measure '{m.column}'")
        agg = {"sum": "SUM", "count": "COUNT", "avg": "AVG", "min": "MIN", "max": "MAX"}.get(m.agg or "")
        expr = f"COUNT(DISTINCT s.\"{col}\")" if m.agg == "count_distinct" else f"{agg}(s.\"{col}\")"
        if m.agg == "count" and col in ("*", ""):
            expr = "COUNT(*)"
        self._date = self.cat.generic_sources[table]
        return table, expr, self.cat.generic_sources[table], None

    def _field_expr(self, f: str, md: MetricDef | None) -> str:
        if f in self.cat.entities:
            if md is not None and f not in md.entities:
                raise CompileError(f"entity '{f}' not available in source {md.source}")
            return f"s.{self.cat.entities[f].key}"
        if f in self.cat.dimensions:
            return self._dim(self.cat.dimensions[f])
        if "." in f:
            return f"s.\"{f.rpartition('.')[2]}\""
        raise CompileError(f"unknown field '{f}'")

    def _where(self, plan: Plan, md: MetricDef | None, date_col: str, time: TimeRange | None,
               drop: set[str] = frozenset()) -> tuple[str, list]:
        conds, params = ["1=1"], []
        if time is not None:
            if not date_col:
                raise CompileError("time filter on a source without date column")
            conds.append(f"CAST(s.\"{date_col}\" AS DATE) BETWEEN ? AND ?")
            params += [time.start, time.end]
        for f in plan.filters:
            if f.field in drop:
                continue
            c, p = self.filter_cond(f, md)
            conds.append(c)
            params += p
        return " AND ".join(conds), params

    def filter_cond(self, f: Filter, md: MetricDef | None) -> tuple[str, list]:
        """One filter over alias s (the metric source)."""
        attr = self._entity_attribute(f.field, md)
        if attr is not None:
            # a negation is the COMPLEMENT ("trừ thuốc kê đơn" = every other product, also those the attribute
            # table does not list), so it is written as NOT IN the positive set, never as a NULL-dropping "!="
            ed, col = attr
            neg = f.op in ("!=", "not_in") or (f.op == "=" and f.value is False)
            pos = f.model_copy(update={"op": {"!=": "=", "not_in": "in"}.get(f.op, f.op),
                                       "value": True if f.value is False else f.value})
            c, p = self._cond(f"d.\"{col}\"", pos, fold=True)
            return (f"s.{ed.key} {'NOT IN' if neg else 'IN'} (SELECT d.{ed.dim_key} FROM {ed.dim_table} d "
                    f"WHERE {c} AND d.{ed.dim_key} IS NOT NULL)"), p
        dd = self.cat.dimensions.get(f.field)
        return self._cond(self._field_expr(f.field, md), f, fold=dd is not None and dd.kind == "string")

    def _entity_attribute(self, f: str, md: MetricDef | None) -> tuple[EntityDef, str] | None:
        """'product.active_ingredient' -> (product entity, column of its dimension table). A bare attribute name
        ('customer_type') resolves to the one entity whose master table has it."""
        e, _, a = f.partition(".")
        if not a and f not in self.cat.entities and f not in self.cat.dimensions:
            owners = [k for k, x in self.cat.entities.items() if x.dim_table and f in x.attributes and k != f]
            if len(owners) != 1:
                return None
            e, a = owners[0], f
        ed = self.cat.entities.get(e)
        if not a or ed is None or not ed.dim_table or a not in ed.attributes:
            return None
        if md is not None and e not in md.entities:
            raise CompileError(f"entity '{e}' not available in source {md.source}")
        return ed, ed.attributes[a]

    @staticmethod
    def _cond(expr: str, f: Filter, fold: bool = False) -> tuple[str, list]:
        """fold=True (attribute values written by people): text equality ignores case, accents and spacing."""
        v = f.value
        vals = v if isinstance(v, list) else [v]
        if fold and f.op in ("=", "!=", "in", "not_in") and vals and all(isinstance(x, str) for x in vals):
            ph = ", ".join("meta.fold_vn(trim(?))" for _ in vals)
            neg = f.op in ("!=", "not_in")
            return f"meta.fold_vn(trim(CAST({expr} AS VARCHAR))) {'NOT IN' if neg else 'IN'} ({ph})", list(vals)
        if f.op in ("in", "not_in"):
            if not vals:
                raise CompileError("empty IN list")
            ph = ", ".join("?" for _ in vals)
            return f"{expr} {'IN' if f.op == 'in' else 'NOT IN'} ({ph})", list(vals)
        if f.op == "between":
            if not (isinstance(v, list) and len(v) == 2):
                raise CompileError("between needs [lo, hi]")
            return f"{expr} BETWEEN ? AND ?", list(v)
        if f.op == "contains":
            return f"CAST({expr} AS VARCHAR) ILIKE ?", [f"%{v}%"]
        if isinstance(v, list):
            if len(v) != 1:
                raise CompileError(f"op {f.op} with a list value")
            v = v[0]
        return f"{expr} {f.op} ?", [v]

    def _group(self, plan: Plan, md: MetricDef | None) -> tuple[str, str, EntityDef | None, DimensionDef | None]:
        e = plan.entity
        if e in self.cat.entities:
            ed = self.cat.entities[e]
            if md is not None and e not in md.entities:
                raise CompileError(f"entity '{e}' not in source {md.source}")
            # most frequent spelling of the code (ties: smallest), deterministic; MIN picked "Ca phe" over "Cà phê"
            return f"s.{ed.key}", f"MODE(s.{ed.label} ORDER BY s.{ed.label})", ed, None
        if e in self.cat.dimensions:
            dd = self.cat.dimensions[e]
            return self._dim(dd), self._dim(dd), None, dd
        raise CompileError(f"unknown entity/dimension '{e}'")

    def _label(self, v: Any, dd: DimensionDef | None, month_only: bool = False) -> Any:
        if dd is None:
            return v
        if dd.kind == "weekday":
            return self.cat.weekday_names.get(int(v), str(v))
        if dd.kind == "date":
            return v.isoformat() if hasattr(v, "isoformat") else str(v)
        if dd.kind == "month":
            return str(v)
        return v

    # ------------------------------------------------------------ execute
    def run(self, plan: Plan, con) -> PlanResult:
        if not plan.answerable:
            return PlanResult("unanswerable", plan.unanswerable_reason)
        fn = getattr(self, f"_op_{plan.op}", None)
        if fn is None:
            raise CompileError(f"op '{plan.op}' is not compiled (custom) -> code solver")
        return fn(plan, con)

    def _scalar(self, con, sql: str, params: list, res: PlanResult) -> Any:
        res.sql.append(sql)
        row = con.execute(sql, params).fetchone()
        return None if row is None else row[0]

    def _op_aggregate(self, plan: Plan, con) -> PlanResult:
        src, expr, dcol, md = self._measure(plan)
        where, params = self._where(plan, md, dcol, plan.time)
        res = PlanResult(plan.answer_type, None)
        v = self._scalar(con, f"SELECT {expr} AS m FROM {src} s WHERE {where}", params, res)
        if v is None:
            res.flags.append("empty_window")
            v = 0 if (md is None or md.additive) else None
        res.value = round_money(v, md.money_decimals) if md is not None else v
        return res

    def _op_exists(self, plan: Plan, con) -> PlanResult:
        src, _, dcol, md = self._measure(plan) if plan.measure else (None, None, None, None)
        if src is None:
            raise CompileError("exists needs a measure to know the source")
        where, params = self._where(plan, md, dcol, plan.time)
        res = PlanResult("boolean", None)
        n = self._scalar(con, f"SELECT COUNT(*) FROM {src} s WHERE {where}", params, res)
        res.value = bool(n)
        return res

    def _ranked(self, plan: Plan, con, limit: int | None) -> tuple[list[tuple], PlanResult, DimensionDef | None]:
        src, expr, dcol, md = self._measure(plan)
        gkey, glabel, ed, dd = self._group(plan, md)
        where, params = self._where(plan, md, dcol, plan.time)
        order = "DESC" if plan.order == "desc" else "ASC"
        if plan.include_zero:
            # universe = master table, so entities with no activity appear with m = 0
            if ed is None or not ed.dim_table:
                raise CompileError("include_zero needs an entity with a master table")
            dkey, dlabel = ed.dim_key or ed.key, ed.dim_label or ed.label
            fill = "0" if (md is None or md.additive) else "NULL"
            base = (f"SELECT d.{dkey} AS k, d.{dlabel} AS label, COALESCE(a.m, {fill}) AS m "
                    f"FROM {ed.dim_table} d LEFT JOIN (SELECT {gkey} AS k, {expr} AS m FROM {src} s "
                    f"WHERE {where} GROUP BY 1) a ON a.k = d.{dkey}")
        else:
            # rows without the entity (walk-in customer, no category) are not a rankable entity
            base = f"SELECT {gkey} AS k, {glabel} AS label, {expr} AS m FROM {src} s WHERE {where} AND {gkey} IS NOT NULL GROUP BY 1"
        sql = f"SELECT * FROM ({base}) t"
        if plan.having is not None:
            c, hparams = self._cond("t.m", plan.having)
            sql += f" WHERE {c}"
            params = params + hparams
        sql += f" ORDER BY t.m {order} NULLS LAST, t.k ASC"
        if limit:
            sql += f" LIMIT {int(limit)}"
        res = PlanResult(plan.answer_type, None)
        res.sql.append(sql)
        rows = con.execute(sql, params).fetchall()
        res.rows, res.rows_preview = rows, rows[:10]
        if not rows:
            res.flags.append("empty_result")
        if limit and len(rows) == limit:
            nxt = con.execute(sql.rsplit(" LIMIT ", 1)[0] + f" LIMIT 1 OFFSET {int(limit)}", params).fetchone()
            if nxt is not None and nxt[2] == rows[-1][2]:
                res.flags.append("tie_at_boundary")   # test set excludes such questions -> suspicious
        return rows, res, dd

    def _emit(self, plan: Plan, rows: list[tuple], res: PlanResult, dd: DimensionDef | None) -> PlanResult:
        labels = [self._label(r[1], dd) for r in rows]
        if plan.answer_type == "string":
            res.value = labels[0] if labels else None
        elif plan.answer_type == "date":
            res.value = labels[0] if labels else None
        elif plan.answer_type in ("number", "percent"):
            res.value = rows[0][2] if rows else None
        elif plan.answer_type == "list":
            res.value = labels
        elif plan.answer_type == "table":
            cols = plan.columns or [plan.entity or "label", (plan.measure.metric or "value") if plan.measure else "value"]
            res.value = {"columns": cols, "rows": [[lab, r[2]] for lab, r in zip(labels, rows)]}
        else:
            raise CompileError(f"answer_type {plan.answer_type} not supported for op {plan.op}")
        return res

    def _op_rank(self, plan: Plan, con) -> PlanResult:
        rows, res, dd = self._ranked(plan, con, plan.k)
        return self._emit(plan, rows, res, dd)

    def _op_argmax_time(self, plan: Plan, con) -> PlanResult:
        if plan.entity not in self.cat.dimensions:
            raise CompileError("argmax_time needs a time dimension entity")
        plan = plan.model_copy(update={"k": plan.k or 1})
        rows, res, dd = self._ranked(plan, con, plan.k)
        return self._emit(plan, rows, res, dd)

    def _op_filter_list(self, plan: Plan, con) -> PlanResult:
        rows, res, dd = self._ranked(plan, con, plan.k)
        return self._emit(plan, rows, res, dd)

    def _op_count(self, plan: Plan, con) -> PlanResult:
        res = PlanResult("number", None)
        if plan.having is not None or plan.include_zero:
            rows, r2, _ = self._ranked(plan.model_copy(update={"k": None}), con, None)
            res.sql, res.value = r2.sql, len(rows)
            return res
        src, _, dcol, md = self._measure(plan)
        key = self._field_expr(plan.entity, md) if plan.entity else "*"
        where, params = self._where(plan, md, dcol, plan.time)
        res.value = self._scalar(con, f"SELECT COUNT(DISTINCT {key}) FROM {src} s WHERE {where}", params, res)
        return res

    def _op_compare(self, plan: Plan, con) -> PlanResult:
        src, expr, dcol, md = self._measure(plan)
        res = PlanResult(plan.answer_type, None)
        vals = []
        for t in (plan.time, plan.compare_time):
            where, params = self._where(plan, md, dcol, t)
            v = self._scalar(con, f"SELECT {expr} FROM {src} s WHERE {where}", params, res)
            vals.append(0 if v is None and (md is None or md.additive) else v)
        a, b = vals
        if a is None or b is None:
            res.flags.append("null_side")
            return res
        mode = plan.compare_mode
        if mode == "diff":
            res.value = a - b
        elif mode == "pct_change":
            if b == 0:
                res.flags.append("division_by_zero")
                res.value = None
            else:
                res.value = (a - b) / b * 100
        elif mode == "ratio":
            res.value = None if b == 0 else a / b
        elif mode == "is_increase":
            res.value = a > b
        elif mode == "is_decrease":
            res.value = a < b
        res.rows_preview = [a, b]
        return res

    def _op_share(self, plan: Plan, con) -> PlanResult:
        src, expr, dcol, md = self._measure(plan)
        res = PlanResult("percent", None)
        where, params = self._where(plan, md, dcol, plan.time)
        num = self._scalar(con, f"SELECT {expr} FROM {src} s WHERE {where}", params, res)
        drop = set(plan.share_drop_fields or ([plan.entity] if plan.entity else [f.field for f in plan.filters]))
        where2, params2 = self._where(plan, md, dcol, plan.time, drop=drop)
        den = self._scalar(con, f"SELECT {expr} FROM {src} s WHERE {where2}", params2, res)
        if not den:
            res.flags.append("division_by_zero")
            return res
        res.value = (num or 0) / den * 100
        res.rows_preview = [num, den]
        return res

    def _op_lookup(self, plan: Plan, con) -> PlanResult:
        ent = next((f for f in plan.filters if f.field in self.cat.entities), None)
        if ent is None or not plan.lookup_attribute:
            raise CompileError("lookup needs an entity filter and lookup_attribute")
        ed = self.cat.entities[ent.field]
        col = ed.attributes.get(plan.lookup_attribute)
        if not (ed.dim_table and col):
            raise CompileError(f"attribute '{plan.lookup_attribute}' unknown for {ed.name}")
        res = PlanResult(plan.answer_type, None)
        v = ent.value[0] if isinstance(ent.value, list) else ent.value
        res.value = self._scalar(con, f"SELECT {col} FROM {ed.dim_table} WHERE {ed.dim_key or ed.key} = ?", [v], res)
        return res
