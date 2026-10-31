"""The `dm` toolkit loaded inside the Code Solver sandbox (05.6 §4).

Few functions, each doing one business job with the rulebook already applied, returning
codes AND names. It reuses the Plan Solver's compiler, so every business rule exists once.
Docstrings are short on purpose: the prompt carries prompts.TOOLKIT_DOC.
"""
from __future__ import annotations

import json
import math
import os
import re
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from dma.context import SplitContext
from dma.nlu.link import link_entities
from dma.rules.parse import parse_rulebook
from dma.semantic.bkp import load_bkp
from dma.solvers.plan.compiler import Compiler
from dma.solvers.plan.dmql import Filter, Measure, Plan, TimeRange
from dma.textnorm import fold, scorer_norm
from dma.timeparse import resolver as R
from dma.timeparse.resolver import Period

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")
BY_ALIASES = {"products": "product", "sp": "product", "san_pham": "product", "item": "product", "mon": "product",
              "customers": "customer", "khach": "customer", "khach_hang": "customer", "categories": "category",
              "nhom": "category", "date": "day", "ngay": "day", "thang": "month", "thu": "weekday", "dow": "weekday",
              "gio": "hour", "nam": "year", "branches": "branch", "store": "branch", "employee": "staff"}


class DM:
    def __init__(self, bkp_dir: str, split_dir: str):
        self.bkp = load_bkp(bkp_dir)
        self.split = SplitContext.load(split_dir)
        self.con = self.bkp.con()
        self.cat = self.bkp.catalog
        self.vindex = self.bkp.vindex
        self.rules = self.split.rulespec
        self.cal = self.split.calendar
        self.trules = self.split.time_rules
        self.weekday_names = self.split.weekday_names
        self._as_of: date | None = None
        self._question = ""
        self._tp = None
        self.calls: list[dict] = []
        self._rb_items = parse_rulebook(self.split.rulebook_text or "")

    # ------------------------------------------------------------ binding
    def _bind_question(self, as_of: str, question: str) -> None:
        self._as_of = date.fromisoformat(str(as_of)[:10])
        self._question = question
        from dma.nlu.analyze import lexicon_for
        from dma.nlu.typos import fix_telex_text
        from dma.textnorm import strip_accents
        plain, _ = fix_telex_text(strip_accents(question), lexicon_for(self.vindex))
        self._tp = R.resolve(plain, self._as_of, self.trules, self.cal)
        self.calls = []

    @property
    def as_of(self) -> date:
        return self._as_of or date.today()

    def _log(self, fn: str, **kw) -> None:
        self.calls.append({"fn": fn, **{k: (v.as_json() if isinstance(v, Period) else v) for k, v in kw.items()}})

    # ------------------------------------------------------------ discovery
    def tables(self) -> pd.DataFrame:
        rows = self.con.cursor().execute(
            "SELECT table_schema, table_name, table_type FROM information_schema.tables "
            "WHERE table_schema IN ('sem', 'typed') AND table_name NOT LIKE '\\_%' ESCAPE '\\' ORDER BY 1, 2").fetchall()
        out = []
        for sch, name, typ in rows:
            n = None
            try:
                n = self.con.cursor().execute(f'SELECT COUNT(*) FROM {sch}."{name}"').fetchone()[0]
            except Exception:  # noqa: BLE001
                pass
            t = self.bkp.bsm.table(name) if sch == "typed" else None
            out.append({"schema": sch, "name": f"{sch}.{name}", "rows": n,
                        "kind": (t.pack_entity or t.grain) if t else ("view" if typ == "VIEW" else "table")})
        return pd.DataFrame(out)

    def schema(self, name: str) -> pd.DataFrame:
        sch, _, tn = name.rpartition(".")
        sch = sch or "typed"
        cols = self.con.cursor().execute(
            "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = ? AND table_name = ? "
            "ORDER BY ordinal_position", [sch, tn]).fetchall()
        roles = {r.column: r.role for r in self.bkp.bsm.roles if r.table == tn} if sch == "typed" else {}
        out = []
        for c, t in cols:
            ex = self.con.cursor().execute(f'SELECT DISTINCT "{c}" FROM {sch}."{tn}" WHERE "{c}" IS NOT NULL LIMIT 3').fetchall()
            out.append({"column": c, "type": t, "role": roles.get(c, ""), "examples": [x[0] for x in ex]})
        return pd.DataFrame(out)

    def sql(self, query: str, params: list | None = None) -> pd.DataFrame:
        """Read-only SQL over sem.* / typed.* (DuckDB dialect)."""
        if re.search(r"\b(insert|update|delete|create|drop|alter|attach|copy|export|install|load|pragma|set)\b", query, re.I):
            raise PermissionError("dm.sql is read-only")
        self._log("sql", query=query[:500])
        return self.con.cursor().execute(query, params or []).df()

    def values(self, column: str, like: str | None = None, k: int = 20) -> pd.DataFrame:
        tbl, _, col = column.rpartition(".")
        tbl = tbl if "." in tbl else (f"typed.{tbl}" if tbl else "sem.sales_lines")
        where, params = "", []
        if like:
            where = f'WHERE meta.fold_vn("{col}") LIKE ?'
            params = [f"%{fold(like)}%"]
        return self.con.cursor().execute(f'SELECT "{col}" AS value, COUNT(*) AS n FROM {tbl} {where} GROUP BY 1 '
                                         f"ORDER BY n DESC, 1 LIMIT {int(k)}", params).df()

    # ------------------------------------------------------------ time
    def period(self, expr: Any) -> Period:
        """Inclusive [start, end] for a Vietnamese time phrase per the rulebook, or a label of dm.periods()."""
        p = self._to_period(expr)
        if p is None:
            raise ValueError(f"không hiểu cụm thời gian {expr!r}. Dùng dm.days('YYYY-MM-DD','YYYY-MM-DD') hoặc dm.month(y, m).")
        self._log("period", expr=str(expr), result=p)
        return p

    def _to_period(self, expr: Any) -> Period | None:
        if expr is None:
            return None
        if isinstance(expr, Period):
            return expr
        if isinstance(expr, dict) and "start" in expr:
            return Period(_d(expr["start"]), _d(expr["end"]), expr.get("label", "custom"))
        if isinstance(expr, (tuple, list)) and len(expr) == 2:
            return Period(_d(expr[0]), _d(expr[1]), "custom")
        if isinstance(expr, (date, datetime, pd.Timestamp)):
            d = _d(expr)
            return Period(d, d, f"day_{d.isoformat()}")
        s = str(expr).strip()
        for p in (self._tp.periods if self._tp else []):
            if s == p.label:
                return p
        m = re.fullmatch(r"(\d{4}-\d{2}-\d{2})\s*(?:\.\.|to|-|đến|den)\s*(\d{4}-\d{2}-\d{2})", s)
        if m:
            return Period(_d(m.group(1)), _d(m.group(2)), "custom")
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            d = _d(s)
            return Period(d, d, f"day_{s}")
        if re.fullmatch(r"\d{4}-\d{2}", s):
            return R.month(int(s[:4]), int(s[5:7]), self.as_of, self.trules)
        tp = R.resolve(s, self.as_of, self.trules, self.cal)
        if tp.periods:
            return tp.periods[0]
        if re.search(r"tu truoc (toi|den) (nay|gio)|toan bo|tat ca|all", fold(s)):
            cov = self.bkp.coverage
            return Period(_d(cov[0]) if cov else date(2000, 1, 1), self.as_of, "all_time")
        return None

    def periods(self) -> list[Period]:
        return list(self._tp.periods) if self._tp else []

    def month(self, y: int, m: int) -> Period:
        return R.month(int(y), int(m), self.as_of, self.trules)

    def quarter(self, y: int, q: int) -> Period:
        return R.quarter(int(y), int(q), self.as_of, self.trules)

    def year(self, y: int) -> Period:
        return R.year(int(y), self.as_of, self.trules)

    def days(self, a: Any, b: Any) -> Period:
        return Period(_d(a), _d(b), "custom")

    def same_period_last_year(self, p: Any) -> Period:
        return R.same_period_last_year(self._to_period(p))

    def tet(self, lunar_year: int) -> Period:
        return R.tet_window(int(lunar_year), self.trules, self.cal)

    def mid_autumn(self, lunar_year: int) -> Period:
        return R.mid_autumn(int(lunar_year), self.trules, self.cal)

    def lunar(self, d: Any):
        return self.cal.lunar(_d(d))

    def weekday_name(self, d: Any) -> str:
        iso = int(d) if isinstance(d, (int, float)) and not isinstance(d, bool) else _d(d).isoweekday()
        return self.weekday_names[iso]

    # ------------------------------------------------------------ entities
    def find(self, text: str, kind: str | None = None, k: int = 5) -> pd.DataFrame:
        """Entity candidates for a mention: [kind, code, name, score, freq] (best first)."""
        rows = []
        for m in link_entities(fold(text), self.vindex, top_k=10, min_score=0.5):
            for c in m.candidates:
                rows.append({"kind": c.kind, "code": c.code or c.name, "name": c.name, "score": c.score, "freq": c.freq})
        if not rows:
            from rapidfuzz import fuzz
            f = fold(text)
            for e in self.vindex.entries:
                s = fuzz.WRatio(f, e.fold) / 100.0
                if s >= 0.6:
                    rows.append({"kind": e.kind, "code": e.code or e.name, "name": e.name, "score": round(s * 0.9, 4), "freq": e.freq})
        df = pd.DataFrame(rows, columns=["kind", "code", "name", "score", "freq"])
        if kind:
            df = df[df["kind"] == kind]
        df = df.sort_values(["score", "freq", "code"], ascending=[False, False, True]).drop_duplicates(["kind", "code"]).head(k)
        self._log("find", text=text, kind=kind, top=df.head(3).to_dict("records"))
        return df.reset_index(drop=True)

    def entity(self, kind: str, code: str) -> dict:
        ed = self.cat.entities.get(kind)
        if ed is None or not ed.dim_table:
            raise ValueError(f"không có bảng master cho '{kind}'. Có: {sorted(self.cat.entities)}")
        code = self._code(kind, code)
        r = self.con.cursor().execute(f"SELECT * FROM {ed.dim_table} WHERE CAST({ed.dim_key or ed.key} AS VARCHAR) = ?",
                                      [str(code)]).df()
        if r.empty:
            raise KeyError(f"{kind} '{code}' không có trong {ed.dim_table}")
        return {k: _py(v) for k, v in r.iloc[0].to_dict().items()}

    def _code(self, kind: str, value: Any) -> Any:
        if isinstance(value, (list, tuple, set)):
            return [self._code(kind, v) for v in value]
        if kind not in self.cat.entities or value is None:
            return value
        c = self.vindex.code_for(kind, str(value))
        return c if c is not None else value

    # ------------------------------------------------------------ business data
    def _filters(self, filters: Any) -> list[Filter]:
        if not filters:
            return []
        if isinstance(filters, list):
            return [f if isinstance(f, Filter) else Filter(**f) for f in filters]
        out = []
        for k, v in dict(filters).items():
            field = BY_ALIASES.get(k, k)
            if field in self.cat.entities:
                v = self._code(field, v)
            if isinstance(v, (list, tuple, set)):
                out.append(Filter(field=field, op="in", value=list(v)))
            else:
                out.append(Filter(field=field, op="=", value=_py(v)))
        return out

    def _tr(self, period: Any) -> TimeRange | None:
        p = self._to_period(period)
        return None if p is None else TimeRange(start=p.start, end=p.end, expr=p.label)

    def _compiler(self) -> Compiler:
        return Compiler(self.cat, as_of=self.as_of)

    def _run(self, plan: Plan):
        res = self._compiler().run(plan, self.con.cursor())
        self._log("query", plan=plan.model_dump(mode="json", exclude_none=True), flags=res.flags, sql=res.sql[-1:] if res.sql else [])
        return res

    def lines(self, period: Any = None, valid: bool = True, **filters) -> pd.DataFrame:
        """Sales lines (valid orders by default) with semantic columns."""
        view = "sem.sales_lines_valid" if valid else "sem.sales_lines"
        conds, params = ["1=1"], []
        p = self._to_period(period)
        if p:
            conds.append("order_date BETWEEN ? AND ?")
            params += [p.start, p.end]
        comp = self._compiler()
        for f in self._filters(filters):
            c, ps = comp.filter_cond(f, None)
            conds.append(c)
            params += ps
        return self.con.cursor().execute(f"SELECT * FROM {view} s WHERE {' AND '.join(conds)} ORDER BY order_date, order_id",
                                         params).df()

    def orders(self, period: Any = None, valid: bool = True) -> pd.DataFrame:
        view = "sem.orders" if valid else "sem.orders_all"
        p = self._to_period(period)
        if p:
            return self.con.cursor().execute(f"SELECT * FROM {view} WHERE order_date BETWEEN ? AND ? ORDER BY order_date, order_id",
                                             [p.start, p.end]).df()
        return self.con.cursor().execute(f"SELECT * FROM {view} ORDER BY order_date, order_id").df()

    def metric(self, name: str, period: Any = None, by: str | None = None, filters: Any = None):
        """One number, or DataFrame[code, name, value] grouped by an entity/dimension."""
        self._check_metric(name)
        if by:
            return self.rank(name, by, period=period, k=None, filters=filters)
        plan = Plan(answer_type="number", op="aggregate", measure=Measure(metric=name), time=self._tr(period),
                    filters=self._filters(filters))
        v = self._run(plan).value
        return _py(v)

    def rank(self, name: str, by: str, period: Any = None, k: int | None = 1, order: str = "desc", filters: Any = None,
             include_zero: bool = False) -> pd.DataFrame:
        """Rank `by` on metric `name` (valid orders). Ties -> code asc. -> DataFrame[code, name, value]."""
        self._check_metric(name)
        by = BY_ALIASES.get(by, by)
        op = "argmax_time" if by in self.cat.dimensions else "rank"
        plan = Plan(answer_type="table", op=op if k else "filter_list", measure=Measure(metric=name), entity=by,
                    k=k, order=order, time=self._tr(period), filters=self._filters(filters), include_zero=include_zero)
        res = self._run(plan)
        dd = self.cat.dimensions.get(by)
        comp = self._compiler()
        rows = [{"code": _py(_iso(r[0])), "name": comp._label(r[1], dd), "value": _py(r[2])} for r in res.rows]
        df = pd.DataFrame(rows, columns=["code", "name", "value"])
        p = self._to_period(period)
        df.attrs["note"] = (f"period {p.start}..{p.end} · " if p else "all time · ") + \
            f"valid orders only · tie-break code asc · flags {res.flags}"
        if res.flags:
            print(f"(flags: {', '.join(res.flags)})")
        return df

    def compare(self, name: str, period_a: Any, period_b: Any, mode: str = "pct_change", filters: Any = None):
        """metric(period_a) vs metric(period_b): pct_change = (a - b) / b * 100."""
        self._check_metric(name)
        plan = Plan(answer_type="percent" if mode == "pct_change" else ("boolean" if mode.startswith("is_") else "number"),
                    op="compare", measure=Measure(metric=name), time=self._tr(period_a), compare_time=self._tr(period_b),
                    compare_mode=mode, filters=self._filters(filters))
        res = self._run(plan)
        if res.value is None and res.flags:
            print(f"(compare undefined: {res.flags}; values {res.rows_preview})")
        else:
            print(f"(a={_py(res.rows_preview[0]) if res.rows_preview else None}, b={_py(res.rows_preview[1]) if len(res.rows_preview) > 1 else None})")
        return _py(res.value)

    def share(self, name: str, part: Any = None, period: Any = None, base: Any = None):
        """part / total * 100 (percent units). `part`, `base` are filter dicts; base defaults to no filter."""
        self._check_metric(name)
        pf = self._filters(part)
        bf = self._filters(base)
        bkeys = {f.field for f in bf}
        drop = [f.field for f in pf if f.field not in bkeys]
        plan = Plan(answer_type="percent", op="share", measure=Measure(metric=name), time=self._tr(period),
                    filters=pf + [f for f in bf if f.field not in {x.field for x in pf}], share_drop_fields=drop or None)
        res = self._run(plan)
        if res.rows_preview:
            print(f"(part={_py(res.rows_preview[0])}, total={_py(res.rows_preview[1])})")
        return _py(res.value)

    def cost(self, product: Any = None, at: Any = None):
        """Average unit cost per product per RuleSpec.cost, using lots purchased up to `at` (default as_of)."""
        at_d = _d(at) if at else self.as_of
        if "sem.purchase_lots" in self.bkp.views and self.rules.cost.method == "fifo":
            raise NotImplementedError("FIFO: dùng dm.sql trên sem.purchase_lots")
        if "sem.product_cost" not in self.bkp.views:
            raise ValueError("Dữ liệu không có giá vốn/giá nhập (gap: cost).")
        df = self.con.cursor().execute("SELECT * FROM sem.product_cost(?) ORDER BY product_id", [at_d]).df()
        if product is not None:
            code = self._code("product", product)
            r = df[df["product_id"] == str(code)]
            return _py(r.iloc[0]["avg_unit_cost"]) if len(r) else None
        return df

    def stock(self, at: Any = None, product: Any = None):
        """Stock on hand at `at` (default as_of) per RuleSpec; one number for a product, else DataFrame."""
        if "stock_on_hand" not in self.cat.metrics:
            raise ValueError("Dữ liệu không có tồn kho (gap: inventory).")
        old = self._as_of
        self._as_of = _d(at) if at else self.as_of
        try:
            if product is not None:
                return self.metric("stock_on_hand", filters={"product": product})
            return self.rank("stock_on_hand", by="product", k=None)
        finally:
            self._as_of = old

    def debt(self, at: Any = None, customer: Any = None, total: bool = False, older_than_days: int | None = None):
        """Outstanding receivable balance at `at` (default as_of): per-customer DataFrame, one customer's number, or
        total=True for the shop total (walk-in debt included). older_than_days=N keeps orders whose debt age
        (days since the ORDER date, today = 0) is > N ("đơn đã đặt quá N ngày")."""
        if "debt_outstanding" not in self.cat.metrics:
            raise ValueError("Dữ liệu không có công nợ (gap: receivables).")
        old = self._as_of
        self._as_of = _d(at) if at else self.as_of
        try:
            period = None
            if older_than_days is not None:
                period = TimeRange(start=date(1900, 1, 1), end=self._as_of - timedelta(days=int(older_than_days) + 1),
                                   expr=f"older than {older_than_days} days")
            if customer is not None:
                return self.metric("debt_outstanding", period=period, filters={"customer": customer})
            if total:
                return self.metric("debt_outstanding", period=period)
            return self.rank("debt_outstanding", by="customer", period=period, k=None)
        finally:
            self._as_of = old

    def expiring(self, within_days: int | None = None, at: Any = None) -> pd.DataFrame:
        """Lots/products whose expiry date is within N days after `at` (rulebook vocabulary if it defines it)."""
        at_d = _d(at) if at else self.as_of
        if within_days is None:
            v = next((e for e in self.rules.vocabulary if "het han" in fold(e.term)), None)
            within_days = int(v.params.get("days", 30)) if v else 30
        for view, extra in (("sem.purchase_lots", "lot_id, quantity"), ("sem.products", "product_name")):
            if view in self.bkp.views:
                df = self.con.cursor().execute(f"SELECT product_id, {extra}, expiry_date FROM {view} WHERE expiry_date "
                                               f"BETWEEN ? AND ? ORDER BY expiry_date, product_id",
                                               [at_d, at_d + timedelta(days=int(within_days))]).df()
                if "expiry_date" in df.columns and len(df) or view == "sem.products":
                    return df
        raise ValueError("Dữ liệu không có hạn sử dụng (gap: expiry).")

    # ------------------------------------------------------------ rules & utils
    def rule(self, path: str) -> dict:
        val = self.rules.get(path)
        ids = self.rules.provenance.get(path, [])
        texts = [it.text for it in self._rb_items if it.item_id in ids]
        return {"value": _py(val), "rulebook": texts}

    def rules_text(self, query: str, k: int = 5) -> list[str]:
        q = set(fold(query).split())
        scored = sorted(((len(q & set(fold(it.text).split())), it.text) for it in self._rb_items), key=lambda x: -x[0])
        return [t for s, t in scored[:k] if s > 0] + [t for t in self.rules.free_text_rules if set(fold(t).split()) & q][:3]

    @staticmethod
    def pct_change(new: Any, old: Any):
        return None if old in (0, None) or new is None else (float(new) - float(old)) / float(old) * 100

    def round2(self, x: Any) -> float:
        from dma.output.formatter import round2
        return round2(x, self.split.format_cfg.rounding)

    def _check_metric(self, name: str) -> None:
        if name not in self.cat.metrics:
            raise ValueError(f"metric '{name}' không có. Có: {', '.join(sorted(self.cat.metrics))}. "
                             f"Thiếu dữ liệu: {', '.join(self.bkp.bsm.gaps) or 'không'}")

    # ------------------------------------------------------------ final answer validation
    def is_data_string(self, s: Any) -> bool:
        s = str(s)
        if self.vindex.is_data_string(s) or scorer_norm(s) in {scorer_norm(v) for v in self.weekday_names.values()}:
            return True
        cur = self.con.cursor()
        for view, col in (("sem.sales_lines", "product_name"), ("sem.sales_lines", "customer_name"),
                          ("sem.sales_lines", "category"), ("sem.sales_lines", "staff_name"), ("sem.sales_lines", "branch_name")):
            try:
                if cur.execute(f"SELECT 1 FROM {view} WHERE lower(trim({col})) = lower(trim(?)) LIMIT 1", [s]).fetchone():
                    return True
            except Exception:  # noqa: BLE001
                continue
        for t in self.bkp.profile.get("tables", []):
            for c in t.get("columns", []):
                if c["sql_type"] != "VARCHAR":
                    continue
                try:
                    if cur.execute(f'SELECT 1 FROM typed."{t["name"]}" WHERE lower(trim("{c["name"]}")) = lower(trim(?)) LIMIT 1',
                                   [s]).fetchone():
                        return True
                except Exception:  # noqa: BLE001
                    continue
        return False

    def _final_check(self, type: str, value: Any, reason: Any, extra: dict) -> tuple[str, Any, dict]:  # noqa: A002
        if type not in ANSWER_TYPES:
            raise ValueError(f"type phải thuộc {ANSWER_TYPES} (abstain do hệ thống tự quyết định)")
        warns = list(extra.get("warnings", []))
        if isinstance(value, pd.Series):
            value = value.tolist()
        if isinstance(value, (list, tuple)) and type in ("number", "percent", "string", "date", "boolean") and len(value) == 1:
            value = value[0]
        value = _py(value)
        if type in ("number", "percent"):
            if isinstance(value, str) or isinstance(value, bool) or value is None:
                raise ValueError("Giá trị số phải là số (int/float), không phải chuỗi/None.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Giá trị số không hữu hạn (chia cho 0?). Kiểm tra lại.")
            if type == "percent" and value != 0 and abs(value) <= 1 and re.search(r"phan tram|%|ty le|ti le|chiem|tang|giam",
                                                                                   fold(self._question) + ("%" if "%" in self._question else "")):
                warns.append("percent_scale_suspect")
        elif type == "string":
            if isinstance(value, (list, tuple)):
                raise ValueError("Kiểu string nhận 1 tên; dùng 'list' cho nhiều tên.")
            if not self.is_data_string(value):
                raise ValueError(f"Tên '{value}' không có trong dữ liệu: copy nguyên văn từ DataFrame (giữ dấu), không tự gõ.")
        elif type == "list":
            if isinstance(value, pd.DataFrame):
                raise ValueError("list nhận danh sách tên/số, không phải DataFrame (dùng df['name'].tolist()).")
            if not isinstance(value, (list, tuple)) or not value:
                raise ValueError("list phải là danh sách không rỗng.")
            value = [_py(v) for v in value]
            bad = [v for v in value if isinstance(v, str) and not self.is_data_string(v)]
            if bad:
                raise ValueError(f"Các tên không có trong dữ liệu: {bad[:3]}")
        elif type == "table":
            if isinstance(value, pd.DataFrame):
                value = {"columns": [str(c) for c in value.columns], "rows": [[_py(x) for x in r] for r in value.itertuples(index=False)]}
            if not (isinstance(value, dict) and "columns" in value and "rows" in value):
                raise ValueError("table phải là DataFrame hoặc {'columns': [...], 'rows': [[...]]}.")
        elif type == "date":
            value = _iso(value) if not isinstance(value, str) else value
        elif type == "boolean":
            if not isinstance(value, bool):
                raise ValueError("boolean phải là True/False.")
        if warns:
            extra = {**extra, "warnings": warns}
        extra = {**extra, "dm_calls": self.calls[-12:]}
        return type, value, extra


def _d(x: Any) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime().date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


def _iso(x: Any) -> Any:
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.date().isoformat() if not (x.hour or x.minute) else x.isoformat()
    if isinstance(x, date):
        return x.isoformat()
    return x


def _py(v: Any) -> Any:
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except ImportError:
        pass
    if isinstance(v, float) and v.is_integer() and abs(v) < 2 ** 53:
        return v
    if hasattr(v, "is_finite") and hasattr(v, "quantize"):   # Decimal
        return float(v)
    if isinstance(v, pd.Timestamp):
        return _iso(v)
    return v


def load_dm(bkp_dir: str, split_dir: str) -> DM:
    return DM(bkp_dir, split_dir)


BOOTSTRAP_BASE = """
import math, json, re, datetime as _dt
import numpy as np, pandas as pd
pd.set_option('display.max_rows', 30); pd.set_option('display.max_columns', 20); pd.set_option('display.width', 200)
from dma.solvers.code.toolkit import load_dm
dm = None
"""


def rebase_code(bkp_dir: str, split_dir: str) -> str:
    return (f"dm = load_dm({bkp_dir!r}, {split_dir!r})\n"
            "__final_check__ = dm._final_check\n")


_ = (json, os)
