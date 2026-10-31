"""Materialize the semantic views sem.* from BSM + RuleSpec (05.2 §5).

Rule-dependent parts (cancelled statuses, return policy, revenue base, cost cutoff) are
parameters, never literals. Missing roles become NULL columns (and gaps)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from dma.ingest.loader import q
from dma.rules.rulespec import RuleSpec
from dma.semantic.bsm import BSM
from dma.textnorm import fold

NULL = "NULL"


def _lit(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


@dataclass
class ViewBuild:
    sql: list[str] = field(default_factory=list)
    views: list[str] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    columns: dict[str, list[str]] = field(default_factory=dict)    # view -> non-null semantic columns


# product attributes beyond the fixed sem.products columns: asked as filters ("thuốc có hoạt chất X", "trừ thuốc kê đơn")
PRODUCT_ATTRIBUTES = ("active_ingredient", "is_prescription", "supplier_name", "reorder_level", "variant", "base_unit",
                      "brand")


def _product_attributes(bsm: BSM, products_t: str, r: dict, pk) -> tuple[dict[str, str], list[str]]:
    """Attribute roles of the product master itself, or of any side table keyed by the same product id/name
    (a drug list next to the product list), joined one row per product."""
    sel: dict[str, str] = {a: f"p.{q(r[a].column)}" for a in PRODUCT_ATTRIBUTES if a in r}
    joins: list[str] = []
    for i, t in enumerate(x.name for x in bsm.tables if x.name != products_t):
        rt = bsm.roles_of(t)
        attrs = [a for a in PRODUCT_ATTRIBUTES if a in rt and a not in sel]
        if not attrs:
            continue
        if "product_id" in rt and "product_id" in r:
            tk, pkey = rt["product_id"].column, r["product_id"].column
            norm = "upper(trim(CAST({} AS VARCHAR)))"
        elif "product_name" in rt and "product_name" in r:
            tk, pkey = rt["product_name"].column, r["product_name"].column
            norm = "meta.fold_vn(trim(CAST({} AS VARCHAR)))"
        else:
            continue
        a = f"pa{i}"
        cols = ", ".join(f"MIN(x.{q(rt[x_].column)}) AS {x_}" for x_ in attrs)
        joins.append(f" LEFT JOIN (SELECT {norm.format('x.' + q(tk))} AS _k, {cols} FROM typed.{q(t)} x GROUP BY 1) {a} "
                     f"ON {a}._k = {norm.format('p.' + q(pkey))}")
        sel.update({x_: f"{a}.{x_}" for x_ in attrs})
    return sel, joins


def _side_product_role(bsm: BSM, products_t: str | None, role: str, key_expr: str | None,
                       name_expr: str | None = None) -> str | None:
    """`role` of a product taken from a side table keyed by the same product id (product list next to a stock
    report chosen as master) or, without ids, by the product name up to spelling; a scalar subquery, or None."""
    for t in bsm.tables:
        if t.name == products_t or t.pack_entity is not None:
            continue
        rt = bsm.roles_of(t.name)
        if role not in rt:
            continue
        if key_expr and "product_id" in rt:
            return (f"(SELECT MIN(x.{q(rt[role].column)}) FROM typed.{q(t.name)} x WHERE upper(trim(CAST(x.{q(rt['product_id'].column)} "
                    f"AS VARCHAR))) = upper(trim(CAST({key_expr} AS VARCHAR))))")
        if name_expr and "product_name" in rt:
            return (f"(SELECT MIN(x.{q(rt[role].column)}) FROM typed.{q(t.name)} x WHERE meta.fold_vn(x.{q(rt['product_name'].column)}) "
                    f"= meta.fold_vn({name_expr}))")
    return None


# placeholders written instead of a customer (folded); README conventions say "Khách lẻ nếu không lưu"
WALK_IN = ("khach le", "kl", "k l", "le", "khach vang lai", "vang lai", "khach qua duong", "khach ghe", "khong ten",
           "khong co", "khach", "khach hang", "walk in", "walk-in", "guest", "-", "--", "n/a", "na", "none", "null")


def _walk_in_sql(name_expr: str) -> str:
    vals = ", ".join(f"'{v}'" for v in WALK_IN)
    return f"(meta.fold_vn({name_expr}) IN ({vals}) OR regexp_matches(meta.fold_vn({name_expr}), '^khach le\\b'))"


class _Src:
    """Resolves a role to a SQL expression over aliases l (lines), o (orders), p (products), c (customers)."""

    def __init__(self, bsm: BSM, lines: str | None, orders: str | None, products: str | None, customers: str | None):
        self.bsm = bsm
        self.alias = {"l": lines, "o": orders, "p": products, "c": customers}

    def ref(self, role: str, order=("l", "o", "p", "c"), mult: bool = False) -> str | None:
        for a in order:
            t = self.alias.get(a)
            if not t:
                continue
            r = self.bsm.roles_of(t).get(role)
            if r:
                e = f"{a}.{q(r.column)}"
                if mult and r.unit_multiplier and r.unit_multiplier != 1.0:
                    e = f"({e} * {r.unit_multiplier:g})"
                return e
        return None

    def role(self, role: str, order=("l", "o", "p", "c")):
        for a in order:
            t = self.alias.get(a)
            if t and role in self.bsm.roles_of(t):
                return a, self.bsm.roles_of(t)[role]
        return None


def _num(e: str | None) -> str:
    return f"CAST({e} AS DOUBLE)" if e else NULL


def _status_case(expr: str, value_map: dict[str, str] | None) -> str:
    if not value_map:
        return "'completed'"
    whens = " ".join(f"WHEN CAST({expr} AS VARCHAR) = {_lit(k)} THEN {_lit(v)}" for k, v in sorted(value_map.items()))
    return f"(CASE WHEN {expr} IS NULL THEN 'completed' {whens} ELSE 'other' END)"


def _note_status(expr: str) -> str:
    from dma.semantic.bsm import load_ontology
    f = f"meta.fold_vn(CAST({expr} AS VARCHAR))"
    whens = " ".join(f"WHEN regexp_matches({f}, {_lit(rx)}) THEN {_lit(lab)}"
                     for lab, rx in (load_ontology().get("status_note_vocabulary") or {}).items())
    return f"(CASE WHEN {expr} IS NULL THEN 'completed' {whens} ELSE 'completed' END)"


def _header_amount_roles(con, bsm: BSM, lt: str, l_order: str | None) -> set[str]:
    """Line gross/discount columns that repeat one value on every line of multi-line orders are order-level
    ("Tiền hàng trước giảm", "Tiền giảm" of the whole invoice): using them per line multiplies them."""
    out: set[str] = set()
    if not l_order:
        return out
    multi = con.execute(f"SELECT COUNT(*) FROM (SELECT 1 FROM typed.{q(lt)} GROUP BY {q(l_order)} "
                        f"HAVING COUNT(*) > 1)").fetchone()[0]
    if not multi:
        return out
    for role in ("line_gross_amount", "line_discount_amount"):
        col = bsm.col(lt, role)
        net = bsm.col(lt, "line_net_amount")
        if not (col and net):
            continue
        varying = con.execute(f"SELECT COUNT(*) FROM (SELECT 1 FROM typed.{q(lt)} GROUP BY {q(l_order)} "
                              f"HAVING COUNT(DISTINCT {q(net)}) > 1 AND COUNT(DISTINCT {q(col)}) = 1 "
                              f"AND MAX({q(col)}) <> 0)").fetchone()[0]
        if varying and header_level(con, lt, l_order, col):
            out.add(role)
    return out


_HONORIFIC = re.compile(r"^((a|anh|chi|c|em|e|co|chu|bac|thay|tho|ong|ba|mr|ms)[\s.]+)+")


def _staff_nicknames(con, bsm: BSM, lines_tables: list[str]) -> dict[str, str]:
    """Books that write staff by nickname ("a Đạt", "A.Đức", "Đức") next to a staff list with full names
    ("Hà Loan Đạt"): raw nickname -> full name, when the given name identifies exactly one person."""
    masters = [t.name for t in bsm.tables if t.name not in lines_tables and bsm.col(t.name, "staff_name")]
    if not masters:
        return {}
    full: list[str] = []
    for m in masters:
        c = bsm.col(m, "staff_name")
        full += [r[0] for r in con.execute(f"SELECT DISTINCT trim(CAST({q(c)} AS VARCHAR)) FROM typed.{q(m)} "
                                           f"WHERE {q(c)} IS NOT NULL").fetchall()]
    raw: set[str] = set()
    for t in lines_tables:
        c = bsm.col(t, "staff_name")
        if c:
            raw |= {r[0] for r in con.execute(f"SELECT DISTINCT trim(CAST({q(c)} AS VARCHAR)) FROM typed.{q(t)} "
                                              f"WHERE {q(c)} IS NOT NULL").fetchall()}
    ffull = {fold(f): f for f in full if f}
    out: dict[str, str] = {}
    for r in raw:
        fr = fold(r)
        if fr in ffull:
            continue
        key = _HONORIFIC.sub("", fr).strip(" .")
        hits = [f for k, f in ffull.items() if key and (k == key or k.endswith(" " + key))]
        if len(set(hits)) == 1:
            out[r] = hits[0]
    return out if len(out) >= max(1, len(raw - set(full)) // 2) else {}


_MONEY_TOK = r"(\d[\d.,]*\s*(?:k|tr\d*|trieu|nghin|ngan|d|dong|vnd)?)"


def _note_money(tok: str) -> float | None:
    from dma.ingest.parsers import VN_FORMAT, parse_number
    t = re.sub(r"\s*(d|dong|vnd)$", "", tok.strip()).replace("trieu", "tr").replace("ngan", "nghin")
    p = parse_number(t, VN_FORMAT) or parse_number(t)
    return float(p.value) if p and p.value is not None else None


def _note_debt(note: str) -> tuple[str, float] | None:
    """('abs', owed) or ('rest', paid: owed = order total - paid) from a free-text debt note, None if unrelated."""
    from dma.textnorm import strip_accents
    f = re.sub(r"\s+", " ", strip_accents(note.lower()).replace("đ", "d"))     # keeps "456.500" (fold drops dots)
    if not re.search(r"\b(no|ghi so|ghi no|thieu|chiu|chua tra|con lai)\b", f):
        return None
    if re.search(r"\b(tra du|ck du|chuyen khoan du|tra het|tra xong|da tra du|du roi|het roi|da thanh toan du)\b", f) \
            and not re.search(r"\b(con thieu|thieu lai|con no)\b", f):
        return ("abs", 0.0)
    m = re.search(r"\b(?:con thieu|thieu lai|thieu|con no|no lai|ghi so|ghi no|no)\s*" + _MONEY_TOK, f)
    if m and _note_money(m.group(1)):
        return ("abs", _note_money(m.group(1)))
    m = re.search(r"\b(?:dua truoc|tra truoc|dua|tra|ck|chuyen khoan)\s*" + _MONEY_TOK + r".*\b(con lai|phan con lai)\b", f)
    if m and _note_money(m.group(1)) is not None:
        return ("rest", _note_money(m.group(1)))
    if re.search(r"\b(no ca don|chua tra|ghi so ca don|ghi no ca don|no het)\b", f):
        return ("rest", 0.0)
    return None


def _note_debt_map(con, bsm: BSM, tables: list[str], note_col: str) -> dict[str, tuple[str, float]]:
    vals: set[str] = set()
    for t in tables:
        if t and note_col in [c[0] for c in con.execute(f"DESCRIBE typed.{q(t)}").fetchall()]:
            vals |= {r[0] for r in con.execute(f"SELECT DISTINCT trim(CAST({q(note_col)} AS VARCHAR)) FROM typed.{q(t)} "
                                               f"WHERE {q(note_col)} IS NOT NULL").fetchall()}
    out = {v: d for v in vals if v and (d := _note_debt(v)) is not None}
    return out if len(out) >= 5 else {}


def decide_amounts(bsm: BSM, src: _Src, prof_tables: dict[str, dict],
                   drop: set[str] | None = None) -> tuple[str, str, str, str]:
    """-> (gross_expr, discount_expr, net_expr, interpretation) per profiled identities (V4/V5)."""
    drop = drop or set()
    qy, pr = _num(src.ref("quantity")), _num(src.ref("unit_price", ("l", "p"), mult=True))
    net = src.ref("line_net_amount", mult=True)
    gross = None if "line_gross_amount" in drop else src.ref("line_gross_amount", mult=True)
    disc = None if "line_discount_amount" in drop else src.ref("line_discount_amount", mult=True)
    pct = src.ref("line_discount_pct")
    tot = src.ref("order_total", ("l",), mult=True)
    has_q, has_p = src.ref("quantity") is not None, src.ref("unit_price", ("l", "p")) is not None
    qp = f"({qy} * {pr})" if has_q and has_p else None
    if net:
        n = _num(net)
        if gross:
            g = _num(gross)
            d = _num(disc) if disc else f"({g} - {n})"
            return g, d, n, "net=column; gross=column"
        if disc:
            d = _num(disc)
            return f"({n} + COALESCE({d}, 0))", d, n, "net=column; gross=net+discount"
        if pct and qp:
            return qp, f"({qp} - {n})", n, "net=column; gross=q*p"
        if qp:
            return qp, f"({qp} - {n})", n, "net=column; gross=q*p"
        return n, "0", n, "net=column (no discount info)"
    if gross:
        g = _num(gross)
        if disc:
            d = _num(disc)
            return g, d, f"({g} - COALESCE({d}, 0))", "net=gross-discount"
        if pct:
            p = _num(pct)
            return g, f"({g} * COALESCE({p}, 0) / 100.0)", f"({g} * (1 - COALESCE({p}, 0) / 100.0))", "net=gross*(1-pct)"
        return g, "0", g, "net=gross (no discount column)"
    if qp:
        if disc:
            d = _num(disc)
            return qp, d, f"({qp} - COALESCE({d}, 0))", "net=q*p-discount"
        if pct:
            p = _num(pct)
            return qp, f"({qp} * COALESCE({p}, 0) / 100.0)", f"({qp} * (1 - COALESCE({p}, 0) / 100.0))", "net=q*p*(1-pct)"
        return qp, "0", qp, "net=q*p"
    if tot:
        t = _num(tot)
        return t, "0", t, "net=order_total (order-grain sales)"
    return NULL, NULL, NULL, "no amount"


def header_level(con, table: str, order_col: str | None, col: str | None) -> bool:
    """True if `col` is constant within each order (header field repeated on lines)."""
    if not (order_col and col):
        return True
    n = con.execute(f"SELECT COUNT(*) FROM (SELECT {q(order_col)} FROM typed.{q(table)} GROUP BY 1 "
                    f"HAVING COUNT(DISTINCT {q(col)}) > 1)").fetchone()[0]
    return n == 0


def build_views(con, bsm: BSM, rs: RuleSpec, prof: dict) -> ViewBuild:
    vb = ViewBuild()
    tmeta = {t["name"]: t for t in prof["tables"]}
    lines_tables = bsm.tables_for("sales_lines")
    orders_t = bsm.table_for("orders")
    products_t = bsm.table_for("products")
    customers_t = bsm.table_for("customers")
    S = vb.sql.append

    def rel(pack: str) -> str | None:
        """typed.<t>, or a UNION ALL BY NAME view when the pack spans same-schema tables (monthly sheets)."""
        ts = bsm.tables_for(pack)
        if not ts:
            return None
        if len(ts) == 1:
            return f"typed.{q(ts[0])}"
        S(f"CREATE OR REPLACE VIEW sem._{pack}_base AS " + " UNION ALL BY NAME ".join(f"SELECT * FROM typed.{q(t)}" for t in ts))
        return f"sem._{pack}_base"

    if lines_tables:
        lt = lines_tables[0]

        def _part(t: str) -> str:
            # rulebook §8: identical rows of the SAME order are an export error (count once);
            # tables not keyed by order keep every row ("mỗi dòng là một lần ghi")
            if bsm.col(t, "order_id"):
                return f"SELECT DISTINCT *, NULL::VARCHAR AS _rid FROM typed.{q(t)}"
            return f"SELECT *, {_lit(t)} || ':' || CAST(rowid AS VARCHAR) AS _rid FROM typed.{q(t)}"
        S("CREATE OR REPLACE VIEW sem._lines_base AS " + " UNION ALL BY NAME ".join(_part(t) for t in lines_tables))
        base = "sem._lines_base"
        if bsm.col(lt, "order_id"):
            vb.notes["dedup"] = "identical rows of the same order counted once"
        has_conv = _unit_conversions(S, bsm, vb)
        src = _Src(bsm, lt, orders_t if orders_t != lt else None, products_t, customers_t)
        joins = []
        l_order = bsm.col(lt, "order_id")
        if orders_t and orders_t != lt and l_order and bsm.col(orders_t, "order_id"):
            joins.append(f"LEFT JOIN {rel('orders')} o ON CAST(o.{q(bsm.col(orders_t, 'order_id'))} AS VARCHAR) = "
                         f"CAST(l.{q(l_order)} AS VARCHAR)")
        else:
            src.alias["o"] = None
        l_prod = bsm.col(lt, "product_id")
        p_key = bsm.col(products_t, "product_id") if products_t else None
        l_pname = bsm.col(lt, "product_name")
        p_name = bsm.col(products_t, "product_name") if products_t else None
        if products_t and l_prod and p_key:
            joins.append(f"LEFT JOIN typed.{q(products_t)} p ON upper(trim(CAST(p.{q(p_key)} AS VARCHAR))) = "
                         f"upper(trim(CAST(l.{q(l_prod)} AS VARCHAR)))")
        elif products_t and l_pname and p_name and not l_prod:
            p_alias = bsm.col(products_t, "product_alias")
            # an alias cell may list several names: "Bỉm Bobby M, tã Bobby M"
            alias_on = (f" OR list_contains(list_transform(string_split(CAST(p.{q(p_alias)} AS VARCHAR), ','), "
                        f"lambda x: meta.fold_vn(trim(x))), meta.fold_vn(trim(l.{q(l_pname)})))" if p_alias else "")
            joins.append(f"LEFT JOIN typed.{q(products_t)} p ON meta.fold_vn(trim(p.{q(p_name)})) = "
                         f"meta.fold_vn(trim(l.{q(l_pname)})){alias_on}")
        else:
            src.alias["p"] = None
        cust_l = src.role("customer_id", ("l", "o"))
        c_key = bsm.col(customers_t, "customer_id") if customers_t else None
        if customers_t and cust_l and c_key:
            a, r = cust_l
            joins.append(f"LEFT JOIN typed.{q(customers_t)} c ON upper(trim(CAST(c.{q(c_key)} AS VARCHAR))) = "
                         f"upper(trim(CAST({a}.{q(r.column)} AS VARCHAR)))")
        else:
            src.alias["c"] = None

        header_amounts = _header_amount_roles(con, bsm, lt, l_order)
        gross, disc, net, interp = decide_amounts(bsm, src, tmeta, drop=header_amounts)
        vb.notes["line_amounts"] = interp
        order_id = f"CAST(l.{q(l_order)} AS VARCHAR)" if l_order else (
            f"CAST(o.{q(bsm.col(orders_t, 'order_id'))} AS VARCHAR)" if src.alias["o"] else "l._rid")
        if not l_order:
            vb.notes["order_id"] = "row id (no order id column: one row = one order)"
        dt = src.ref("order_datetime", ("l", "o"))
        d = src.ref("order_date", ("l", "o"))
        tm = src.ref("order_time", ("l", "o"))
        date_expr = f"CAST({d} AS DATE)" if d else (f"CAST({dt} AS DATE)" if dt else NULL)
        ts_expr = f"CAST({dt} AS TIMESTAMP)" if dt else (
            f"TRY_CAST(CAST({date_expr} AS VARCHAR) || ' ' || CAST({tm} AS VARCHAR) AS TIMESTAMP)" if tm and d else NULL)
        dr = src.role("order_date", ("l", "o"))
        if ts_expr == NULL and dr:
            tname = src.alias[dr[0]]
            dcol = next((c for c in tmeta.get(tname, {}).get("columns", []) if c["name"] == dr[1].column), {})
            if dcol.get("has_time"):
                ts_expr = f"CAST({d} AS TIMESTAMP)"     # one datetime column carries both date and time of day
        p_joined_by_name = bool(src.alias["p"] and not l_prod and p_key)
        # without a product code, a product IS its name up to spelling (case, accents, spacing): rulebook §3 logic
        p_alias_join = bool(src.alias["p"] and not l_prod and not p_key and p_name and bsm.col(products_t, "product_alias"))
        pid = f"CAST(l.{q(l_prod)} AS VARCHAR)" if l_prod else (
            f"COALESCE(CAST(p.{q(p_key)} AS VARCHAR), meta.fold_vn(l.{q(l_pname)}))" if p_joined_by_name else
            f"meta.fold_vn(COALESCE(CAST(p.{q(p_name)} AS VARCHAR), l.{q(l_pname)}))" if p_alias_join else
            f"meta.fold_vn(l.{q(l_pname)})" if l_pname else (
                f"meta.fold_vn({src.ref('service_name', ('l',))})" if src.ref("service_name", ("l",)) else NULL))
        pname_parts = [e for e in (src.ref("product_name", ("p",)), src.ref("product_name", ("l",)),
                                   src.ref("service_name", ("l",))) if e]
        pname = f"COALESCE({', '.join(f'CAST({e} AS VARCHAR)' for e in pname_parts)})" if pname_parts else pid
        cat_parts = [e for e in (src.ref("category", ("p",)), src.ref("category", ("l",))) if e]
        if not cat_parts:
            cat_parts = [e for e in (_side_product_role(bsm, products_t, "category", f"l.{q(l_prod)}" if l_prod else None,
                                                        f"l.{q(l_pname)}" if l_pname else None),) if e]
        category = f"COALESCE({', '.join(cat_parts)})" if cat_parts else NULL
        st = src.role("order_status", ("l", "o"))
        vmap = dict(st[1].value_map or {}) if st else {}
        if st:                        # unioned monthly sheets: each sheet saw its own status values
            for t in (lines_tables if st[0] == "l" else bsm.tables_for("orders"))[1:]:
                r2 = bsm.roles_of(t).get("order_status")
                if r2 and r2.value_map:
                    vmap.update({k: v for k, v in r2.value_map.items() if k not in vmap})
        status = _status_case(f"{st[0]}.{q(st[1].column)}", vmap) if st else "'completed'"
        note = None if st else src.ref("note", ("l", "o"))
        if note:   # rulebook §8: a book without a status column writes the status in its note; no status word = completed
            status = _note_status(note)
        ls = src.role("line_status", ("l",))
        ret_parts = []
        if ls:
            ret_parts.append(f"({_status_case(f'l.{q(ls[1].column)}', ls[1].value_map)} = 'returned')")
        rq = src.ref("returned_qty", ("l",))
        if rq:
            ret_parts.append(f"(COALESCE(CAST({rq} AS DOUBLE), 0) > 0)")
        qline = src.ref("quantity", ("l",))
        if qline:   # returned goods written as negative quantities on the same order
            ret_parts.append(f"(COALESCE(TRY_CAST({qline} AS DOUBLE), 0) < 0)")
        rf = src.ref("returned_flag", ("l",))
        if rf:
            ret_parts.append(f"(COALESCE(TRY_CAST({rf} AS BOOLEAN), FALSE))")
        returned = " OR ".join(ret_parts) if ret_parts else "FALSE"

        def ent(id_role, name_role, master_alias=None):
            idx = src.ref(id_role, ("l", "o"))
            names = [e for e in ((src.ref(name_role, (master_alias,)) if master_alias else None),
                                 src.ref(name_role, ("l", "o"))) if e]
            nm = f"COALESCE({', '.join(f'CAST({e} AS VARCHAR)' for e in names)})" if names else None
            i = f"CAST({idx} AS VARCHAR)" if idx else (f"trim({nm})" if nm else NULL)
            return i, nm or (i if idx else NULL)

        cid, cname = ent("customer_id", "customer_name", "c" if src.alias["c"] else None)
        # rulebook §3: a walk-in order ("Khách lẻ", "kl", blank) has NO customer (not counted, not ranked);
        # without a customer code, names that differ only in writing (case, accents, spacing) are one customer
        if cid != NULL:
            walk = _walk_in_sql(cname if cname != NULL else cid)
            # "An 0327988481", "An - 0327 988 481", "An (0327988481)": same name and number, only written differently
            key = cid if src.ref("customer_id", ("l", "o")) else f"regexp_replace(meta.fold_vn({cid}), '[^a-z0-9]+', '', 'g')"
            cid = f"CASE WHEN {walk} OR NULLIF(trim({cid}), '') IS NULL THEN NULL ELSE {key} END"
            if cname != NULL:
                cname = f"CASE WHEN {walk} THEN NULL ELSE {cname} END"
        bid, bname = ent("branch_id", "branch_name")
        sid, sname = ent("staff_id", "staff_name")
        if not src.ref("staff_id", ("l", "o")) and sname != NULL:
            nick = _staff_nicknames(con, bsm, lines_tables)
            if nick:
                whens = " ".join(f"WHEN {_lit(k)} THEN {_lit(v)}" for k, v in sorted(nick.items()))
                sname = f"COALESCE(CASE trim(CAST({sname} AS VARCHAR)) {whens} END, {sname})"
                sid = f"trim({sname})"
        ord_table = orders_t if src.alias["o"] else lt
        ord_alias = "o" if src.alias["o"] else "l"

        def order_level(role):
            r = src.role(role, ("o", "l"))
            if not r:
                return NULL, True
            a, cr = r
            e = f"CAST({a}.{q(cr.column)}{f' * {cr.unit_multiplier:g}' if cr.unit_multiplier != 1.0 else ''} AS DOUBLE)"
            hl = True if a == "o" else header_level(con, lt, l_order, cr.column)
            return e, hl

        # ---- quantities in BASE units (rulebook: "số lượng quy về đơn vị cơ bản")
        line_unit = src.ref("unit", ("l",))
        qb = src.ref("quantity_base", ("l",))
        fpn, fpid = f"meta.fold_vn(trim(CAST({pname} AS VARCHAR)))", f"meta.fold_vn(trim(CAST({pid} AS VARCHAR)))"
        if qb:
            qty = _num(qb)
            vb.notes["quantity"] = "base units from the converted-quantity column"
        elif has_conv and qline and line_unit:
            joins.append(f"LEFT JOIN sem.unit_conv cv ON cv.unit_f = meta.fold_vn(trim(CAST({line_unit} AS VARCHAR))) "
                         f"AND cv.pkey_f IN ({fpid}, {fpn})")
            qty = f"({_num(qline)} * COALESCE(cv.factor, 1.0))"
            vb.notes["quantity"] = "line quantity x conversion factor (unit table / README)"
        else:
            qty = _num(qline)
        base_unit_parts = [e for e in (src.ref("base_unit", ("p",)), src.ref("base_unit", ("l",))) if e]
        if has_conv:
            base_unit_parts.append(f"(SELECT MIN(u.base_unit) FROM sem.unit_conv u WHERE u.pkey_f IN ({fpid}, {fpn}))")
        if src.ref("unit", ("p",)) and line_unit:
            base_unit_parts.append(src.ref("unit", ("p",)))
        if line_unit and not qb:
            base_unit_parts.append(f"CASE WHEN {'cv.factor IS NULL' if has_conv and qline else 'TRUE'} THEN {line_unit} END")
        base_unit = f"COALESCE({', '.join(f'CAST({e} AS VARCHAR)' for e in base_unit_parts)})" if base_unit_parts else NULL
        # ---- lines that are not goods (service fee, order-level voucher/points, shipping) -> no product, no quantity
        kind = _line_kind_expr(pname)
        # an order is dated by its sale: a return line written on the return day ("dòng trả hàng: lúc trả") keeps the
        # order in the period it was placed (rulebook: "theo ngày đặt"); the line's own date stays in line_date
        if order_id != "l._rid" and date_expr != NULL:
            o_date, o_ts = (f"MIN({date_expr}) OVER (PARTITION BY {order_id})",
                            f"MIN({ts_expr}) OVER (PARTITION BY {order_id})" if ts_expr != NULL else NULL)
        else:
            o_date, o_ts = date_expr, ts_expr
        cols = {
            "order_id": order_id, "order_date": o_date, "order_ts": o_ts, "line_date": date_expr,
            "product_id": f"CASE WHEN {kind} = 'item' THEN {pid} END",
            "product_name": pname, "category": category, "line_kind": kind,
            "variant": src.ref("variant", ("l",)) or NULL,
            "unit": line_unit or src.ref("unit", ("p",)) or NULL,
            "base_unit": base_unit,
            "quantity_line": _num(qline),
            "quantity": f"CASE WHEN {kind} = 'item' THEN {qty} END",
            "unit_price": _num(src.ref("unit_price", ("l", "p"), mult=True)),
            "line_gross_amount": gross, "line_discount_amount": disc, "line_net_amount": net,
            "customer_id": cid, "customer_name": cname, "customer_type": src.ref("customer_type", ("l", "o", "c")) or NULL,
            "branch_id": bid, "branch_name": bname, "staff_id": sid, "staff_name": sname,
            "channel": src.ref("channel", ("l", "o")) or NULL,
            "payment_method": src.ref("payment_method", ("l", "o")) or NULL,
            "table_no": src.ref("table_no", ("l", "o")) or NULL,
            "order_status": status, "line_returned": f"({returned})",
            "unit_cost_line": _num(src.ref("cost_price", ("l",), mult=True)),
            "list_cost_price": _num(src.ref("cost_price", ("p",), mult=True)),
        }
        hdr = {}
        for role, out in (("voucher_amount", "order_voucher"), ("vat_amount", "order_vat"),
                          ("shipping_fee", "order_shipping"), ("order_total", "order_total"),
                          ("paid_amount", "order_paid"), ("debt_amount", "order_debt")):
            e, hl = order_level(role)
            cols[out] = e
            hdr[out] = hl
        note_ref = src.ref("note", ("l", "o"))
        if cols["order_debt"] == NULL and note_ref and cols["order_total"] != NULL:
            nd = _note_debt_map(con, bsm, lines_tables if note_ref.startswith("l.") else [orders_t], note_ref.split(".", 1)[1].strip('"'))
            if nd:   # README: "đơn ghi nợ thì ghi khách đã đưa / còn thiếu bao nhiêu" in the order note
                whens = " ".join(f"WHEN {_lit(k)} THEN " + (f"{v[1]}" if v[0] == "abs" else
                                                             f"GREATEST({cols['order_total']} - {v[1]}, 0)")
                                 for k, v in sorted(nd.items()))
                cols["order_debt"] = f"(CASE trim(CAST({note_ref} AS VARCHAR)) {whens} ELSE NULL END)"
                hdr["order_debt"] = True
                vb.notes["order_debt"] = "parsed from the order note (paid / still owed amounts written in words)"
        # revenue per RuleSpec.revenue (line-level part)
        rv = rs.revenue
        qxp = f"({cols['quantity']} * {cols['unit_price']})" \
            if cols["quantity"] != NULL and cols["unit_price"] != NULL else net
        rev_base = {"line_net": net, "line_gross": gross, "qty_x_price": qxp}[rv.base]
        keep = ["'item'"] + (["'service_fee'"] if rv.include_service_fee else []) + \
            (["'shipping'"] if rv.include_shipping else []) + (["'order_discount'"] if rv.subtract_order_voucher else [])
        rev_line = f"(CASE WHEN {kind} IN ({', '.join(keep)}) THEN {rev_base} ELSE 0 END)"
        order_adj = []
        if rv.include_vat and cols["order_vat"] != NULL:
            if hdr["order_vat"]:
                order_adj.append("+ COALESCE(MAX(order_vat), 0)")
            else:
                rev_line = f"({rev_line} + COALESCE({cols['order_vat']}, 0))"
        if rv.include_shipping and cols["order_shipping"] != NULL:
            if hdr["order_shipping"]:
                order_adj.append("+ COALESCE(MAX(order_shipping), 0)")
            else:
                rev_line = f"({rev_line} + COALESCE({cols['order_shipping']}, 0))"
        if rv.subtract_order_voucher and cols["order_voucher"] != NULL:
            if hdr["order_voucher"]:
                order_adj.append("- COALESCE(MAX(order_voucher), 0)")
            else:
                rev_line = f"({rev_line} - COALESCE({cols['order_voucher']}, 0))"
        cols["revenue_amount"] = rev_line
        vb.notes["revenue"] = f"line={rv.base}; order_adjustments={' '.join(order_adj) or 'none'}"
        vb.notes["order_level_revenue"] = "1" if order_adj else "0"
        select = ",\n  ".join(f"{e} AS {k}" for k, e in cols.items())
        S(f"CREATE OR REPLACE VIEW sem.sales_lines AS SELECT\n  {select}\nFROM {base} l\n" + "\n".join(joins))
        vb.views.append("sem.sales_lines")
        vb.columns["sem.sales_lines"] = [k for k, e in cols.items() if e != NULL]
        canceled = ", ".join(_lit(s) for s in rs.valid_order.canceled_statuses) or "'canceled'"
        S("CREATE OR REPLACE VIEW sem.order_flags AS SELECT order_id, "
          f"BOOL_OR(order_status IN ({canceled})) AS is_canceled, "
          "BOOL_OR(line_returned OR order_status = 'returned') AS has_return FROM sem.sales_lines GROUP BY order_id")
        cond = ["NOT f.is_canceled"]
        if rs.valid_order.exclude_returned_lines_only:
            cond.append("NOT s.line_returned AND s.order_status <> 'returned'")
        elif rs.valid_order.exclude_order_if_any_line_returned:
            cond.append("NOT f.has_return")
        S("CREATE OR REPLACE VIEW sem.sales_lines_valid AS SELECT s.* FROM sem.sales_lines s "
          f"JOIN sem.order_flags f USING (order_id) WHERE {' AND '.join(cond)}")
        vb.views += ["sem.order_flags", "sem.sales_lines_valid"]
        agg_hdr = {k: (f"MAX({k})" if hdr[k] else f"SUM({k})") for k in hdr}
        okeys = ("customer_id", "customer_name", "branch_id", "staff_id", "channel", "payment_method")
        for vname, srcv in (("sem.orders", "sem.sales_lines_valid"), ("sem.orders_all", "sem.sales_lines")):
            S(f"CREATE OR REPLACE VIEW {vname} AS SELECT order_id, MIN(order_date) AS order_date, MIN(order_ts) AS order_ts, "
              + ", ".join(f"MIN({k}) AS {k}" for k in okeys)
              + ", MIN(order_status) AS order_status, SUM(line_net_amount) AS lines_net_amount, "
              + "SUM(quantity) AS quantity, "
              + ", ".join(f"{agg_hdr[k]} AS {k}" for k in hdr)
              + f", SUM(revenue_amount) {' '.join(order_adj)} AS revenue_amount"
              + f" FROM {srcv} GROUP BY order_id")
            vb.views.append(vname)
        _ = ord_table, ord_alias

    # ------------------------------------------------------------ masters
    if products_t:
        r = bsm.roles_of(products_t)
        pk = r.get("product_id") or r.get("product_name")

        def pc(role, mult=False):
            x = r.get(role)
            if not x:
                return NULL
            e = f"p.{q(x.column)}"
            return f"CAST({e} * {x.unit_multiplier:g} AS DOUBLE)" if mult and x.unit_multiplier != 1.0 else e
        attr_sel, attr_joins = _product_attributes(bsm, products_t, r, pk)
        S("CREATE OR REPLACE VIEW sem.products AS SELECT "
          f"{('CAST(p.' + q(pk.column) + ' AS VARCHAR)') if r.get('product_id') else ('meta.fold_vn(p.' + q(pk.column) + ')')} AS product_id, "
          f"CAST({pc('product_name') if r.get('product_name') else 'p.' + q(pk.column)} AS VARCHAR) AS product_name, "
          f"{pc('category') if r.get('category') or not r.get('product_id') else (_side_product_role(bsm, products_t, 'category', 'p.' + q(pk.column)) or NULL)} AS category, {pc('unit')} AS unit, {_num(pc('list_price', True)) if r.get('list_price') else NULL} AS list_price, "
          f"{_num(pc('cost_price', True)) if r.get('cost_price') else NULL} AS cost_price, {pc('expiry_date')} AS expiry_date, "
          f"{pc('supplier_id')} AS supplier_id, {_num(pc('qty_on_hand')) if r.get('qty_on_hand') else NULL} AS qty_on_hand"
          + "".join(f", {e} AS {a}" for a, e in attr_sel.items())
          + f" FROM typed.{q(products_t)} p" + "".join(attr_joins))
        vb.columns["sem.products"] = list(attr_sel)
    elif lines_tables:
        S("CREATE OR REPLACE VIEW sem.products AS SELECT product_id, MODE(product_name ORDER BY product_name) AS product_name, MIN(category) AS category, "
          "MIN(unit) AS unit, NULL::DOUBLE AS list_price, MIN(list_cost_price) AS cost_price, NULL::DATE AS expiry_date, "
          "NULL::VARCHAR AS supplier_id, NULL::DOUBLE AS qty_on_hand FROM sem.sales_lines WHERE product_id IS NOT NULL GROUP BY 1")
    if products_t or lines_tables:
        vb.views.append("sem.products")
        S("CREATE OR REPLACE VIEW sem.categories AS SELECT DISTINCT category FROM ("
          "SELECT category FROM sem.products" + (" UNION ALL SELECT category FROM sem.sales_lines" if lines_tables else "")
          + ") WHERE category IS NOT NULL")
        vb.views.append("sem.categories")
    if customers_t:
        r = bsm.roles_of(customers_t)
        ck = r.get("customer_id") or r.get("customer_name")
        nm = r.get("customer_name") or ck
        ckey = f"CAST(c.{q(ck.column)} AS VARCHAR)" if r.get("customer_id") else f"meta.fold_vn(c.{q(ck.column)})"
        S(f"CREATE OR REPLACE VIEW sem.customers AS SELECT {ckey} AS customer_id, "
          f"CAST(c.{q(nm.column)} AS VARCHAR) AS customer_name, "
          f"{('c.' + q(r['customer_type'].column)) if r.get('customer_type') else NULL} AS customer_type "
          f"FROM typed.{q(customers_t)} c")
        vb.views.append("sem.customers")
    elif lines_tables and "customer_id" in vb.columns.get("sem.sales_lines", []):
        S("CREATE OR REPLACE VIEW sem.customers AS SELECT customer_id, MODE(customer_name ORDER BY customer_name) AS customer_name, "
          "MIN(customer_type) AS customer_type FROM sem.sales_lines WHERE customer_id IS NOT NULL GROUP BY 1")
        vb.views.append("sem.customers")
    for ent, cols_ in (("branches", ("branch_id", "branch_name")), ("staff", ("staff_id", "staff_name")),
                       ("channels", ("channel", "channel")), ("payment_methods", ("payment_method", "payment_method"))):
        if lines_tables and cols_[0] in vb.columns.get("sem.sales_lines", []):
            S(f"CREATE OR REPLACE VIEW sem.{ent} AS SELECT {cols_[0]} AS code, MODE({cols_[1]} ORDER BY {cols_[1]}) AS name "
              f"FROM sem.sales_lines WHERE {cols_[0]} IS NOT NULL GROUP BY 1")
            vb.views.append(f"sem.{ent}")

    # ------------------------------------------------------------ purchasing / cost
    pt = bsm.table_for("purchase_lots")
    if pt:
        r = bsm.roles_of(pt)

        def rc(role, mult=False):
            x = r.get(role)
            if not x:
                return None
            e = f"t.{q(x.column)}"
            return f"({e} * {x.unit_multiplier:g})" if mult and x.unit_multiplier != 1.0 else e
        pid = rc("product_id") or (f"meta.fold_vn({rc('product_name')})" if rc("product_name") else None)
        if pid and not rc("product_id") and products_t and bsm.col(products_t, "product_name") and bsm.col(products_t, "product_id"):
            pid = (f"COALESCE((SELECT MIN(CAST(pp.{q(bsm.col(products_t, 'product_id'))} AS VARCHAR)) "
                   f"FROM typed.{q(products_t)} pp WHERE meta.fold_vn(pp.{q(bsm.col(products_t, 'product_name'))}) = "
                   f"meta.fold_vn({rc('product_name')})), {pid})")
        qty = rc("purchase_qty")
        cost = rc("unit_cost", True)
        amt = rc("purchase_amount", True)
        if cost is None and amt and qty:
            cost = f"({amt} / NULLIF({qty}, 0))"
        # receipt header (date, supplier) of receipt lines, joined by the receipt id
        hdr_join, hdate, hsup = "", None, None
        if r.get("purchase_id") and not r.get("purchase_date"):
            ht = next((x.name for x in bsm.tables if x.name != pt and bsm.col(x.name, "purchase_id")
                       and bsm.col(x.name, "purchase_date")), None)
            if ht:
                hdr_join = (f" LEFT JOIN (SELECT CAST({q(bsm.col(ht, 'purchase_id'))} AS VARCHAR) AS _pid, "
                            f"MIN({q(bsm.col(ht, 'purchase_date'))}) AS _pdate"
                            + (f", MIN({q(bsm.col(ht, 'supplier_id'))}) AS _sup" if bsm.col(ht, "supplier_id") else "")
                            + f" FROM typed.{q(ht)} GROUP BY 1) h ON h._pid = CAST(t.{q(r['purchase_id'].column)} AS VARCHAR)")
                hdate, hsup = "h._pdate", ("h._sup" if bsm.col(ht, "supplier_id") else None)
        pdate = rc("purchase_date") or hdate
        if pid and qty and pdate:
            S("CREATE OR REPLACE VIEW sem.purchase_lots AS SELECT "
              f"CAST({pid} AS VARCHAR) AS product_id, CAST({pdate} AS DATE) AS purchase_date, "
              f"{rc('lot_id') or NULL} AS lot_id, CAST({qty} AS DOUBLE) AS quantity, "
              f"CAST({cost or 'NULL'} AS DOUBLE) AS unit_cost, "
              f"{rc('expiry_date') or NULL} AS expiry_date, {rc('supplier_id') or hsup or NULL} AS supplier_id "
              f"FROM {rel('purchase_lots')} t{hdr_join}")
            vb.views.append("sem.purchase_lots")
            if not cost:
                vb.notes["purchase_lots"] = "receipt lines without prices: stock and expiry only, no purchase cost"
        else:
            vb.warnings.append("purchase table without product/qty/date roles -> no purchase views")
    if "sem.purchase_lots" in vb.views and "sem.order_flags" in vb.views:
        # sales-by-lot table carrying a returned quantity ("Hàng khách trả về lại lô đã bán")
        lt_ = next((x.name for x in bsm.tables if all(bsm.col(x.name, k) for k in ("lot_id", "order_id", "returned_qty"))), None)
        if lt_:
            S("CREATE OR REPLACE VIEW sem.lot_returns AS SELECT pl.product_id, o.order_date, "
              f"CAST(x.{q(bsm.col(lt_, 'returned_qty'))} AS DOUBLE) AS returned_qty FROM typed.{q(lt_)} x "
              f"JOIN (SELECT lot_id, MIN(product_id) AS product_id FROM sem.purchase_lots GROUP BY 1) pl "
              f"ON CAST(pl.lot_id AS VARCHAR) = CAST(x.{q(bsm.col(lt_, 'lot_id'))} AS VARCHAR) "
              f"JOIN (SELECT order_id, MIN(order_date) AS order_date FROM sem.sales_lines GROUP BY 1) o "
              f"ON o.order_id = CAST(x.{q(bsm.col(lt_, 'order_id'))} AS VARCHAR) "
              f"JOIN sem.order_flags f ON f.order_id = o.order_id "
              f"WHERE NOT f.is_canceled AND x.{q(bsm.col(lt_, 'returned_qty'))} IS NOT NULL")
            vb.views.append("sem.lot_returns")
    if "sem.purchase_lots" in vb.views and rs.cost.method != "static_cost_price" and \
            "no purchase cost" not in vb.notes.get("purchase_lots", ""):
        if rs.cost.method == "latest_cost":
            S("CREATE OR REPLACE MACRO sem.product_cost(cutoff) AS TABLE SELECT product_id, "
              "arg_max(unit_cost, purchase_date) AS avg_unit_cost FROM sem.purchase_lots "
              "WHERE purchase_date <= CAST(cutoff AS DATE) GROUP BY product_id")
        else:   # weighted average by lot (FIFO approximated by the same until a FIFO table exists)
            S("CREATE OR REPLACE MACRO sem.product_cost(cutoff) AS TABLE SELECT product_id, "
              "SUM(quantity * unit_cost) / NULLIF(SUM(quantity), 0) AS avg_unit_cost FROM sem.purchase_lots "
              "WHERE purchase_date <= CAST(cutoff AS DATE) GROUP BY product_id")
            if rs.cost.method == "fifo":
                vb.warnings.append("cost.method=fifo approximated by weighted average in SQL; dm.cost implements FIFO")
        S("CREATE OR REPLACE VIEW sem.cost_curve AS WITH d AS (SELECT product_id, purchase_date, SUM(quantity) q, "
          "SUM(quantity * unit_cost) c FROM sem.purchase_lots GROUP BY ALL) SELECT product_id, purchase_date, "
          "SUM(c) OVER w / NULLIF(SUM(q) OVER w, 0) AS avg_cost_to_date FROM d "
          "WINDOW w AS (PARTITION BY product_id ORDER BY purchase_date ROWS UNBOUNDED PRECEDING)")
        vb.views += ["sem.product_cost", "sem.cost_curve"]
        vb.notes["cost"] = rs.cost.method
    elif (products_t and bsm.col(products_t, "cost_price")) or (lines_tables and src_has(vb, "unit_cost_line")):
        S("CREATE OR REPLACE MACRO sem.product_cost(cutoff) AS TABLE SELECT product_id, MIN(cost_price) AS avg_unit_cost "
          "FROM sem.products WHERE cost_price IS NOT NULL GROUP BY product_id")
        vb.views.append("sem.product_cost")
        vb.notes["cost"] = "static_cost_price"

    # ------------------------------------------------------------ inventory
    it = bsm.table_for("inventory")
    if it:
        r = bsm.roles_of(it)
        pidc = r.get("product_id") or r.get("product_name")
        if pidc and r.get("qty_on_hand"):
            dcol = r.get("snapshot_date")
            pid = f"CAST(t.{q(pidc.column)} AS VARCHAR)" if r.get("product_id") else f"meta.fold_vn(t.{q(pidc.column)})"
            if not r.get("product_id") and "sem.products" in vb.views:
                # stock list written by product NAME: use the catalog code the sales lines use
                pid = (f"COALESCE((SELECT MIN(pp.product_id) FROM sem.products pp WHERE meta.fold_vn(pp.product_name) = "
                       f"meta.fold_vn(t.{q(pidc.column)})), {pid})")
            S(f"CREATE OR REPLACE VIEW sem.inventory AS SELECT {pid} AS product_id, "
              f"{('CAST(t.' + q(dcol.column) + ' AS DATE)') if dcol else 'NULL::DATE'} AS snapshot_date, "
              f"CAST(t.{q(r['qty_on_hand'].column)} AS DOUBLE) AS qty_on_hand FROM {rel('inventory')} t")
            vb.views.append("sem.inventory")
    elif products_t and bsm.col(products_t, "qty_on_hand"):
        S("CREATE OR REPLACE VIEW sem.inventory AS SELECT product_id, NULL::DATE AS snapshot_date, qty_on_hand FROM sem.products")
        vb.views.append("sem.inventory")

    # ------------------------------------------------------------ receivables
    rt = bsm.table_for("receivables")
    if rt:
        r = bsm.roles_of(rt)
        cust = r.get("customer_id") or r.get("customer_name")
        date_ = r.get("ledger_date")
        amount = None
        if r.get("debit_amount") or r.get("credit_amount"):
            dbt = f"COALESCE(CAST(t.{q(r['debit_amount'].column)} AS DOUBLE), 0)" if r.get("debit_amount") else "0"
            crd = f"COALESCE(CAST(t.{q(r['credit_amount'].column)} AS DOUBLE), 0)" if r.get("credit_amount") else "0"
            amount = f"({dbt} - {crd})"
            ltype = "CASE WHEN " + amount + " >= 0 THEN 'debit' ELSE 'credit' END"
        elif r.get("ledger_type") and r.get("ledger_amount"):
            lt_ = r["ledger_type"]
            ltype = _status_case(f"t.{q(lt_.column)}", lt_.value_map).replace("'completed'", "'other'")
            a = f"CAST(t.{q(r['ledger_amount'].column)} AS DOUBLE)"
            amount = f"(CASE WHEN {ltype} = 'credit' THEN -ABS({a}) WHEN {ltype} = 'debit' THEN ABS({a}) ELSE 0 END)"
        if cust and amount:
            cexpr = f"CAST(t.{q(cust.column)} AS VARCHAR)"
            ckey = f"CASE WHEN {_walk_in_sql(cexpr)} THEN NULL ELSE " + \
                (f"{cexpr} END" if r.get("customer_id") else f"meta.fold_vn({cexpr}) END")
            S(f"CREATE OR REPLACE VIEW sem.receivables AS SELECT {ckey} AS customer_id, "
              f"{('CAST(t.' + q(date_.column) + ' AS DATE)') if date_ else 'NULL::DATE'} AS ledger_date, "
              f"{ltype} AS ledger_type, {amount} AS signed_amount FROM {rel('receivables')} t")
            vb.views.append("sem.receivables")
    elif _payments_debt(bsm, vb, rel, S):
        vb.views += ["sem.payments", "sem.debt_orders", "sem.receivables"]
        vb.notes["receivables"] = "orders (not canceled, not COD/marketplace) minus payments dated up to the cutoff"
    elif lines_tables and ("order_debt" in vb.columns.get("sem.sales_lines", []) or
                           ("order_total" in vb.columns.get("sem.sales_lines", []) and "order_paid" in vb.columns.get("sem.sales_lines", []))):
        amt = "order_debt" if "order_debt" in vb.columns["sem.sales_lines"] else "(order_total - order_paid)"
        S(f"CREATE OR REPLACE VIEW sem.receivables AS SELECT customer_id, order_date AS ledger_date, 'debit' AS ledger_type, "
          f"COALESCE({amt}, 0) AS signed_amount FROM sem.orders")
        vb.views.append("sem.receivables")
        vb.notes["receivables"] = f"from orders: {amt}"
    for s in vb.sql:
        con.execute(s)
    return vb


NOT_DEBT_PAYMENT = r"\bcod\b|thu ho|giao hang thu tien"          # rulebook §4: COD / marketplace orders are not debt
NOT_DEBT_CHANNEL = r"shopee|lazada|tiki|tiktok|sendo|san tmdt|san thuong mai|marketplace"


def _payments_debt(bsm: BSM, vb: ViewBuild, rel, S) -> bool:
    """Receivables from a payment book: debt of an order at X = order total - payments dated <= X."""
    pt = bsm.table_for("payments")
    cols = vb.columns.get("sem.sales_lines", [])
    if not pt or "sem.orders_all" not in vb.views or not ({"order_total", "line_net_amount"} & set(cols)):
        return False
    r = bsm.roles_of(pt)
    date_r = r.get("payment_date") or r.get("order_datetime") or r.get("order_date")
    if not (r.get("order_id") and r.get("ledger_amount") and date_r):
        return False
    S(f"CREATE OR REPLACE VIEW sem.payments AS SELECT CAST(t.{q(r['order_id'].column)} AS VARCHAR) AS order_id, "
      f"CAST(t.{q(date_r.column)} AS DATE) AS pay_date, CAST(t.{q(r['ledger_amount'].column)} AS DOUBLE) AS amount, "
      f"{('CAST(t.' + q(r['payment_method'].column) + ' AS VARCHAR)') if r.get('payment_method') else 'NULL::VARCHAR'} "
      f"AS payment_method FROM {rel('payments')} t WHERE t.{q(r['ledger_amount'].column)} IS NOT NULL")
    total = "o.order_total" if "order_total" in cols else (
        "(o.lines_net_amount" + (" + COALESCE(o.order_shipping, 0)" if "order_shipping" in cols else "")
        + (" - COALESCE(o.order_voucher, 0)" if "order_voucher" in cols else "") + ")")
    excl = []
    if "payment_method" in cols:
        excl.append(f"regexp_matches(meta.fold_vn(COALESCE(o.payment_method, '')), '{NOT_DEBT_PAYMENT}')")
    if "channel" in cols:
        excl.append(f"regexp_matches(meta.fold_vn(COALESCE(o.channel, '')), '{NOT_DEBT_CHANNEL}')")
    S("CREATE OR REPLACE VIEW sem.debt_orders AS SELECT o.order_id, o.customer_id, o.customer_name, o.order_date, "
      f"CAST({total} AS DOUBLE) AS order_total, "
      "COALESCE((SELECT SUM(p.amount) FROM sem.payments p WHERE p.order_id = o.order_id AND p.pay_date <= o.order_date), 0) "
      "AS paid_at_purchase FROM sem.orders_all o JOIN sem.order_flags f USING (order_id) "
      f"WHERE NOT f.is_canceled{''.join(' AND NOT ' + e for e in excl)}")
    S("CREATE OR REPLACE MACRO sem.debt_at(x) AS TABLE SELECT d.order_id, d.customer_id, d.customer_name, d.order_date, "
      "d.order_total - COALESCE((SELECT SUM(p.amount) FROM sem.payments p WHERE p.order_id = d.order_id "
      "AND p.pay_date <= CAST(x AS DATE)), 0) AS remaining FROM sem.debt_orders d WHERE d.order_date <= CAST(x AS DATE)")
    # ledger form kept for generic code: debit = order total on the order day, credit = each payment
    S("CREATE OR REPLACE VIEW sem.receivables AS SELECT customer_id, order_date AS ledger_date, 'debit' AS ledger_type, "
      "order_total AS signed_amount, order_id FROM sem.debt_orders UNION ALL SELECT d.customer_id, p.pay_date, 'credit', "
      "-p.amount, d.order_id FROM sem.payments p JOIN sem.debt_orders d USING (order_id)")
    return True


def _line_kind_expr(pname: str) -> str:
    from dma.semantic.bsm import load_ontology
    pats = load_ontology().get("non_item_lines") or {}
    f = f"meta.fold_vn(trim(CAST({pname} AS VARCHAR)))"
    whens = " ".join(f"WHEN regexp_matches({f}, {_lit(rx)}) THEN {_lit(k)}" for k, rx in pats.items())
    return f"(CASE {whens} ELSE 'item' END)" if whens else "'item'"


def _unit_conversions(S, bsm: BSM, vb: ViewBuild) -> bool:
    """sem.unit_conv(pkey_f, unit_f, factor, base_unit) from conversion tables and README lists."""
    parts = []
    for t in bsm.tables_for("unit_conversions"):
        r = bsm.roles_of(t)
        key = r.get("product_id") or r.get("product_name")
        if not (key and r.get("conv_factor") and (r.get("big_unit") or r.get("base_unit"))):
            continue
        bu = f"CAST(c.{q(r['base_unit'].column)} AS VARCHAR)" if r.get("base_unit") else "NULL::VARCHAR"
        unit = r.get("big_unit") or r.get("base_unit")
        parts.append(f"SELECT meta.fold_vn(trim(CAST(c.{q(key.column)} AS VARCHAR))) AS pkey_f, "
                     f"meta.fold_vn(trim(CAST(c.{q(unit.column)} AS VARCHAR))) AS unit_f, "
                     f"CAST(c.{q(r['conv_factor'].column)} AS DOUBLE) AS factor, {bu} AS base_unit FROM typed.{q(t)} c")
    if bsm.unit_conversions:
        S("CREATE OR REPLACE TABLE meta.readme_conv(product VARCHAR, big_unit VARCHAR, factor DOUBLE, base_unit VARCHAR)")
        vals = ", ".join(f"({_lit(p)}, {_lit(u)}, {float(f)}, {_lit(b)})" for p, u, f, b in bsm.unit_conversions)
        S(f"INSERT INTO meta.readme_conv VALUES {vals}")
        parts.append("SELECT meta.fold_vn(trim(product)) AS pkey_f, meta.fold_vn(trim(big_unit)) AS unit_f, factor, "
                     "base_unit FROM meta.readme_conv")
    if not parts:
        return False
    S("CREATE OR REPLACE VIEW sem.unit_conv AS SELECT pkey_f, unit_f, MIN(factor) AS factor, MIN(base_unit) AS base_unit "
      "FROM (" + " UNION ALL ".join(parts) + ") WHERE factor IS NOT NULL GROUP BY 1, 2")
    vb.views.append("sem.unit_conv")
    return True


def src_has(vb: ViewBuild, col: str) -> bool:
    return col in vb.columns.get("sem.sales_lines", [])
