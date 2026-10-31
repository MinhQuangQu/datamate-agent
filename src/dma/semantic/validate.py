"""Deterministic BSM validation V1–V9 (05.2 §4.3) on the materialized views.
Result: warnings + bsm_quality (share of applicable checks passed) for the calibrator."""
from __future__ import annotations

from dma.semantic.bsm import BSM, dtype_compatible, load_ontology


def sanitize_roles(bsm: BSM, prof: dict) -> list[str]:
    """V1: drop roles on unknown columns or with incompatible dtypes (in place)."""
    ont = load_ontology()["roles"]
    cols = {(t["name"], c["name"]): c for t in prof["tables"] for c in t.get("columns", [])}
    keep, warns, seen = [], [], set()
    for r in bsm.roles:
        c = cols.get((r.table, r.column))
        if c is None:
            warns.append(f"V1 drop {r.table}.{r.column}->{r.role}: unknown column")
            continue
        if r.role == "order_time" and c["kind"] == "datetime":
            # a "Giờ" column holding full timestamps is the order datetime (hour of day still comes from it)
            r = r.model_copy(update={"role": "order_datetime"})
        spec = ont.get(r.role)
        if spec and not dtype_compatible(spec["dtype"], c["kind"]) and not (spec["dtype"] == "date" and c["kind"] == "datetime"):
            warns.append(f"V1 drop {r.table}.{r.column}->{r.role}: {c['kind']} incompatible with {spec['dtype']}")
            continue
        if (r.table, r.role) in seen and not r.role.startswith("custom:"):
            continue
        seen.add((r.table, r.role))
        keep.append(r)
    bsm.roles = keep
    return warns


def _one(con, sql: str):
    try:
        row = con.execute(sql).fetchone()
        return None if row is None else row[0]
    except Exception:  # noqa: BLE001
        return None


def validate(con, bsm: BSM, views: list[str], view_cols: dict[str, list[str]], tol: float = 1.0) -> tuple[list[str], float, dict]:
    checks: dict[str, bool | None] = {}
    warns: list[str] = []
    v = set(views)
    cols = set(view_cols.get("sem.sales_lines", []))
    if "sem.products" in v:
        n, d = _one(con, "SELECT COUNT(*) FROM sem.products"), _one(con, "SELECT COUNT(DISTINCT product_id) FROM sem.products")
        checks["V2_products_key_unique"] = bool(n) and d / n >= 0.995
    if "sem.customers" in v:
        n, d = _one(con, "SELECT COUNT(*) FROM sem.customers"), _one(con, "SELECT COUNT(DISTINCT customer_id) FROM sem.customers")
        checks["V2_customers_key_unique"] = bool(n) and d / n >= 0.995
    if "sem.sales_lines" in v:
        if "sem.products" in v and bsm.table_for("products"):
            cov = _one(con, "SELECT AVG(CASE WHEN p.product_id IS NOT NULL THEN 1.0 ELSE 0 END) FROM "
                            "(SELECT DISTINCT product_id FROM sem.sales_lines) s LEFT JOIN sem.products p USING (product_id)")
            checks["V3_fk_products"] = cov is not None and cov >= 0.95
        if {"quantity", "unit_price"} <= cols:
            share = _one(con, f"SELECT AVG(CASE WHEN ABS(quantity * unit_price - line_gross_amount) <= {tol} "
                              f"OR ABS(quantity * unit_price - line_net_amount - line_discount_amount) <= {tol} THEN 1.0 ELSE 0 END) "
                              "FROM sem.sales_lines WHERE quantity IS NOT NULL AND unit_price IS NOT NULL "
                              "AND line_gross_amount IS NOT NULL")
            checks["V4_qty_x_price"] = share is None or share >= 0.95
        share = _one(con, f"SELECT AVG(CASE WHEN ABS(line_gross_amount - COALESCE(line_discount_amount, 0) - line_net_amount) "
                          f"<= {tol} THEN 1.0 ELSE 0 END) FROM sem.sales_lines WHERE line_net_amount IS NOT NULL")
        checks["V5_gross_minus_discount"] = share is None or share >= 0.95
        other = _one(con, "SELECT AVG(CASE WHEN order_status = 'other' THEN 1.0 ELSE 0 END) FROM sem.sales_lines")
        checks["V6_status_mapped"] = other is None or other <= 0.01
        if other and other > 0.01:
            warns.append(f"V6: {other:.1%} of lines have an unmapped status")
        bad = _one(con, "SELECT COUNT(*) FROM (SELECT order_id FROM sem.sales_lines GROUP BY 1 "
                        "HAVING COUNT(DISTINCT order_date) > 1)")
        tot = _one(con, "SELECT COUNT(DISTINCT order_id) FROM sem.sales_lines")
        checks["V7_header_consistent"] = not tot or (bad or 0) / tot <= 0.01
        if "quantity" in cols:
            neg = _one(con, "SELECT AVG(CASE WHEN quantity < 0 AND NOT line_returned THEN 1.0 ELSE 0 END) FROM sem.sales_lines")
            checks["V8_qty_nonneg"] = neg is None or neg <= 0.01
        rng = con.execute("SELECT MIN(order_date), MAX(order_date) FROM sem.sales_lines").fetchone()
        checks["V9_date_range"] = rng[0] is None or (rng[0].year >= 2015 and rng[1].year <= 2035)
        nulls = _one(con, "SELECT AVG(CASE WHEN order_date IS NULL THEN 1.0 ELSE 0 END) FROM sem.sales_lines")
        if nulls and nulls > 0.01:
            warns.append(f"{nulls:.1%} sales lines without a date")
        nn = _one(con, "SELECT AVG(CASE WHEN line_net_amount IS NULL THEN 1.0 ELSE 0 END) FROM sem.sales_lines")
        if nn and nn > 0.01:
            warns.append(f"{nn:.1%} sales lines without an amount")
    if "unit_cost_line" in view_cols.get("sem.sales_lines", []):
        # cost written on the sale lines (the rulebook's gross-profit source): those lines must carry it
        cov = _one(con, "SELECT AVG(CASE WHEN unit_cost_line IS NOT NULL THEN 1.0 ELSE 0 END) FROM sem.sales_lines "
                        "WHERE line_kind = 'item'")
        checks["V10_cost_coverage"] = cov is None or cov >= 0.95
        if cov is not None and cov < 0.95:
            warns.append(f"only {cov:.0%} of sold lines carry a cost: profit answers are unreliable")
    elif "sem.product_cost" in v and "sem.sales_lines" in v:
        cov = _one(con, "SELECT AVG(CASE WHEN c.product_id IS NOT NULL THEN 1.0 ELSE 0 END) FROM "
                        "(SELECT DISTINCT product_id FROM sem.sales_lines) s LEFT JOIN sem.product_cost(DATE '9999-12-31') c "
                        "USING (product_id)")
        checks["V10_cost_coverage"] = cov is None or cov >= 0.95
        if cov is not None and cov < 0.95:
            warns.append(f"only {cov:.0%} of sold products have a purchase cost: profit/cost answers are unreliable")
    for k, ok in checks.items():
        if ok is False:
            warns.append(f"{k} failed")
    applicable = [x for x in checks.values() if x is not None]
    quality = sum(bool(x) for x in applicable) / len(applicable) if applicable else 0.5
    return warns, round(quality, 3), checks
