"""Metric catalog from BSM + RuleSpec + materialized views (05.2 §6, 05.5 §4).
A metric whose roles are missing is NOT registered -> DMQL using it fails to compile,
which is evidence for `unanswerable`."""
from __future__ import annotations

import json
from dataclasses import asdict

from dma.rules.rulespec import RuleSpec
from dma.semantic.bsm import BSM
from dma.semantic.views import ViewBuild
from dma.solvers.plan.compiler import Catalog, DimensionDef, EntityDef, MetricDef, default_dimensions

DEFAULT_WEEKDAYS = {1: "Thứ Hai", 2: "Thứ Ba", 3: "Thứ Tư", 4: "Thứ Năm", 5: "Thứ Sáu", 6: "Thứ Bảy", 7: "Chủ nhật"}

METRIC_DESC = {
    "revenue": "doanh thu theo rulebook (đơn hợp lệ)", "quantity": "tổng số lượng bán (đơn hợp lệ)",
    "order_count": "số đơn hợp lệ", "order_count_all": "số đơn kể cả hủy/trả (dùng với filter order_status)",
    "customer_count": "số khách (đơn hợp lệ)", "avg_order_value": "giá trị trung bình mỗi đơn",
    "discount_amount": "tổng chiết khấu dòng", "avg_unit_price": "giá bán bình quân = doanh thu/số lượng",
    "line_count": "số dòng hàng hợp lệ", "gross_amount": "tổng tiền hàng trước chiết khấu",
    "cogs": "giá vốn hàng bán", "gross_profit": "lợi nhuận gộp = doanh thu - giá vốn", "margin_pct": "biên lợi nhuận %",
    "cancel_rate": "% đơn bị hủy trên tổng đơn", "canceled_order_count": "số đơn bị hủy",
    "return_rate": "% đơn có trả hàng", "returned_order_count": "số đơn có trả hàng",
    "debt_outstanding": "công nợ còn lại đến as_of (đơn còn nợ > 0)", "debtor_count": "số khách còn nợ đến as_of",
    "debt_added": "nợ thêm: phần chưa trả trong ngày mua của đơn đặt trong kỳ",
    "debt_collected": "thu nợ: tiền trả sau ngày mua có ngày trả trong kỳ", "stock_on_hand": "tồn kho tại as_of",
    "purchase_amount": "tổng tiền nhập", "purchase_qty": "tổng số lượng nhập",
    "voucher_amount": "tổng voucher cấp đơn", "vat_amount": "tổng VAT", "shipping_fee": "tổng phí giao hàng",
}

MONEY_METRICS = {"revenue", "revenue_lines", "avg_order_value", "avg_unit_price", "discount_amount", "gross_amount",
                 "cogs", "gross_profit", "debt_outstanding", "debt_added", "debt_collected", "purchase_amount",
                 "voucher_amount", "vat_amount", "shipping_fee"}


def build_catalog(bsm: BSM, rs: RuleSpec, vb: ViewBuild, weekday_names: dict[int, str] | None = None,
                  generic_tables: dict[str, str] | None = None) -> Catalog:
    cols = set(vb.columns.get("sem.sales_lines", []))
    views = set(vb.views)
    ents: dict[str, EntityDef] = {}
    mets: dict[str, MetricDef] = {}
    if "sem.sales_lines" in views:
        if "product_id" in cols:
            ents["product"] = EntityDef("product", "product_id", "product_name", "sem.products", "product_id",
                                        "product_name", {"list_price": "list_price", "unit": "unit", "category": "category",
                                                         "cost_price": "cost_price", "expiry_date": "expiry_date",
                                                         "supplier_id": "supplier_id", "qty_on_hand": "qty_on_hand",
                                                         "product_name": "product_name",
                                                         **{a: a for a in vb.columns.get("sem.products", [])}})
        if "category" in cols:
            ents["category"] = EntityDef("category", "category", "category", "sem.categories", "category", "category")
        if "customer_id" in cols:
            ents["customer"] = EntityDef("customer", "customer_id", "customer_name",
                                         "sem.customers" if "sem.customers" in views else None, "customer_id",
                                         "customer_name", {"customer_type": "customer_type", "customer_name": "customer_name"})
        for e, key, label, dim in (("branch", "branch_id", "branch_name", "sem.branches"),
                                   ("staff", "staff_id", "staff_name", "sem.staff"),
                                   ("channel", "channel", "channel", "sem.channels"),
                                   ("payment_method", "payment_method", "payment_method", "sem.payment_methods")):
            if key in cols:
                ents[e] = EntityDef(e, key, label, dim if dim in views else None, "code", "name")
        line_ents = tuple(ents)
        V, ALL = "sem.sales_lines_valid", "sem.sales_lines"
        order_ents = tuple(e for e in line_ents if e in ("customer", "branch", "staff", "channel", "payment_method"))
        if vb.notes.get("order_level_revenue") == "1":
            mets["revenue"] = MetricDef("revenue", "sem.orders", "SUM(s.revenue_amount)", "order_date", order_ents,
                                        description=METRIC_DESC["revenue"] + " (cấp đơn: VAT/ship/voucher)")
            mets["revenue_lines"] = MetricDef("revenue_lines", V, "SUM(s.revenue_amount)", "order_date", line_ents,
                                              description="doanh thu cấp dòng (để xếp hạng theo sản phẩm)")
        else:
            mets["revenue"] = MetricDef("revenue", V, "SUM(s.revenue_amount)", "order_date", line_ents,
                                        description=METRIC_DESC["revenue"])
        if "quantity" in cols:
            mets["quantity"] = MetricDef("quantity", V, "SUM(s.quantity)", "order_date", line_ents)
            mets["avg_unit_price"] = MetricDef("avg_unit_price", V, "SUM(s.revenue_amount) / NULLIF(SUM(s.quantity), 0)",
                                               "order_date", line_ents, additive=False)
        mets["order_count"] = MetricDef("order_count", V, "COUNT(DISTINCT s.order_id)", "order_date", line_ents)
        mets["order_count_all"] = MetricDef("order_count_all", ALL, "COUNT(DISTINCT s.order_id)", "order_date", line_ents)
        mets["line_count"] = MetricDef("line_count", V, "COUNT(*)", "order_date", line_ents)
        if "customer_id" in cols:
            mets["customer_count"] = MetricDef("customer_count", V, "COUNT(DISTINCT s.customer_id)", "order_date", line_ents)
        mets["avg_order_value"] = MetricDef("avg_order_value", "sem.orders",
                                            "SUM(s.revenue_amount) / NULLIF(COUNT(DISTINCT s.order_id), 0)", "order_date",
                                            order_ents, additive=False)
        if vb.notes.get("line_amounts", "").find("discount") >= 0 or "line_discount_amount" in cols:
            mets["discount_amount"] = MetricDef("discount_amount", V, "SUM(s.line_discount_amount)", "order_date", line_ents)
            mets["gross_amount"] = MetricDef("gross_amount", V, "SUM(s.line_gross_amount)", "order_date", line_ents)
        canceled = ", ".join("'" + s.replace("'", "''") + "'" for s in rs.valid_order.canceled_statuses) or "'canceled'"
        mets["canceled_order_count"] = MetricDef(
            "canceled_order_count", ALL, f"COUNT(DISTINCT CASE WHEN s.order_status IN ({canceled}) THEN s.order_id END)",
            "order_date", line_ents)
        mets["cancel_rate"] = MetricDef(
            "cancel_rate", ALL, f"COUNT(DISTINCT CASE WHEN s.order_status IN ({canceled}) THEN s.order_id END) * 100.0 "
                                "/ NULLIF(COUNT(DISTINCT s.order_id), 0)", "order_date", line_ents, additive=False)
        mets["returned_order_count"] = MetricDef(
            "returned_order_count", ALL, "COUNT(DISTINCT CASE WHEN s.line_returned OR s.order_status = 'returned' "
                                         "THEN s.order_id END)", "order_date", line_ents)
        mets["return_rate"] = MetricDef(
            "return_rate", ALL, "COUNT(DISTINCT CASE WHEN s.line_returned OR s.order_status = 'returned' THEN s.order_id END)"
                                " * 100.0 / NULLIF(COUNT(DISTINCT s.order_id), 0)", "order_date", line_ents, additive=False)
        for key, col in (("voucher_amount", "order_voucher"), ("vat_amount", "order_vat"),
                         ("shipping_fee", "order_shipping")):
            if col in cols:
                mets[key] = MetricDef(key, "sem.orders", f"SUM(s.{col})", "order_date", order_ents)
        # cost of goods sold -> gross profit (05.5 §4)
        cost_src = None
        qty_col = "quantity"
        if rs.cost.profit_source == "sale_line" and "unit_cost_line" in cols:
            # rulebook: cost of a sale line = quantity x the cost written on that line, per SOLD unit
            cost_src = f"(SELECT v.*, v.unit_cost_line AS avg_unit_cost FROM {V} v)"
            qty_col = "quantity_line" if "quantity_line" in cols else "quantity"
        elif "sem.product_cost" in views and "product_id" in cols:
            cut = rs.cost.lots_cutoff
            if cut == "until_sale_date" and "sem.cost_curve" in views:
                cost_src = (f"(SELECT v.*, c.avg_cost_to_date AS avg_unit_cost FROM {V} v ASOF LEFT JOIN sem.cost_curve c "
                            f"ON v.product_id = c.product_id AND v.order_date >= c.purchase_date)")
            else:
                cutoff = "DATE '9999-12-31'" if cut == "all" else "$as_of"
                cost_src = (f"(SELECT v.*, c.avg_unit_cost FROM {V} v LEFT JOIN sem.product_cost({cutoff}) c "
                            f"USING (product_id))")
        elif "unit_cost_line" in cols:
            cost_src = f"(SELECT v.*, v.unit_cost_line AS avg_unit_cost FROM {V} v)"
        if cost_src and "quantity" in cols:
            # non-additive: NULL (cost unknown for the sold products) must stay NULL, never become 0
            # all sold lines must have a cost, otherwise the value is unknown (NULL), not a partial sum
            item = " AND s.line_kind = 'item'" if "line_kind" in cols else ""   # fees/vouchers carry no cost
            ok = f"COUNT(*) FILTER (WHERE s.avg_unit_cost IS NULL{item}) = 0"
            cogs = f"CASE WHEN {ok} THEN SUM(COALESCE(s.{qty_col} * s.avg_unit_cost, 0)) END"
            mets["cogs"] = MetricDef("cogs", cost_src, cogs, "order_date", line_ents, additive=False)
            mets["gross_profit"] = MetricDef("gross_profit", cost_src, f"SUM(s.revenue_amount) - ({cogs})", "order_date",
                                             line_ents, additive=False)
            mets["margin_pct"] = MetricDef("margin_pct", cost_src,
                                           f"(SUM(s.revenue_amount) - ({cogs})) * 100.0 / NULLIF(SUM(s.revenue_amount), 0)",
                                           "order_date", line_ents, additive=False)
    if "sem.receivables" in views and "customer" not in ents:
        ents["customer"] = EntityDef("customer", "customer_id", "customer_id",
                                     "sem.customers" if "sem.customers" in views else None, "customer_id", "customer_name")
    if "sem.debt_orders" in views:
        # rulebook §4: an order's debt at X counts only when > 0; walk-in debt is in the total but has no customer row
        mets["debt_outstanding"] = MetricDef("debt_outstanding", "(SELECT * FROM sem.debt_at($as_of) WHERE remaining > 0)",
                                             "SUM(s.remaining)", "order_date", ("customer",))
        mets["debtor_count"] = MetricDef("debtor_count", "(SELECT * FROM sem.debt_at($as_of) WHERE remaining > 0)",
                                         "COUNT(DISTINCT s.customer_id)", "order_date", ("customer",), additive=False)
        mets["debt_added"] = MetricDef("debt_added", "(SELECT d.*, d.order_total - d.paid_at_purchase AS added "
                                       "FROM sem.debt_orders d WHERE d.order_total - d.paid_at_purchase > 0)",
                                       "SUM(s.added)", "order_date", ("customer",))
        mets["debt_collected"] = MetricDef("debt_collected", "(SELECT p.*, d.customer_id, d.customer_name, d.order_date "
                                           "FROM sem.payments p JOIN sem.debt_orders d USING (order_id) "
                                           "WHERE p.pay_date > d.order_date)", "SUM(s.amount)", "pay_date", ("customer",))
    elif "sem.receivables" in views:
        mets["debt_outstanding"] = MetricDef("debt_outstanding", "(SELECT r.*, COALESCE(cu.customer_name, r.customer_id) AS customer_name "
                                             "FROM sem.receivables r LEFT JOIN "
                                             + ("sem.customers" if "sem.customers" in views else
                                                "(SELECT NULL::VARCHAR AS customer_id, NULL::VARCHAR AS customer_name)")
                                             + " cu USING (customer_id) WHERE r.ledger_date IS NULL OR r.ledger_date <= $as_of)",
                                             "SUM(s.signed_amount)", "ledger_date", ("customer",))
    if "sem.inventory" in views:
        if "product" not in ents:
            ents["product"] = EntityDef("product", "product_id", "product_id", None)
        src = ("(SELECT i.*, pr.product_name, pr.category FROM sem.inventory i LEFT JOIN "
               + ("sem.products" if "sem.products" in views else "(SELECT NULL::VARCHAR product_id, NULL::VARCHAR product_name, NULL::VARCHAR category)")
               + " pr USING (product_id) WHERE i.snapshot_date IS NULL OR i.snapshot_date = (SELECT MAX(snapshot_date) "
               "FROM sem.inventory i2 WHERE i2.product_id = i.product_id AND i2.snapshot_date <= $as_of))")
        mets["stock_on_hand"] = MetricDef("stock_on_hand", src, "SUM(s.qty_on_hand)", "",
                                          tuple(e for e in ("product", "category") if e in ents))
    elif "sem.purchase_lots" in views and "sem.sales_lines" in views:
        # goods physically left the shop on every NOT canceled order (a returned order kept what it did not
        # return): received − sold + returned. Negative return lines net out in the sum; returned quantities
        # written in a separate column (sales-by-lot table) come from sem.lot_returns.
        ret = ("UNION ALL SELECT product_id, returned_qty FROM sem.lot_returns WHERE order_date <= $as_of "
               if "sem.lot_returns" in views else "")
        src = ("(SELECT t.product_id, MIN(pr.product_name) AS product_name, MIN(pr.category) AS category, SUM(t.q) AS qty_on_hand "
               "FROM (SELECT product_id, quantity AS q FROM sem.purchase_lots WHERE purchase_date <= $as_of "
               "UNION ALL SELECT s.product_id, -s.quantity FROM sem.sales_lines s JOIN sem.order_flags f USING (order_id) "
               "WHERE NOT f.is_canceled AND s.line_kind = 'item' AND s.order_date <= $as_of " + ret + ") t "
               "LEFT JOIN sem.products pr USING (product_id) GROUP BY t.product_id)")
        mets["stock_on_hand"] = MetricDef("stock_on_hand", src, "SUM(s.qty_on_hand)", "",
                                          tuple(e for e in ("product", "category") if e in ents),
                                          description="tồn = Σ nhập − Σ xuất (đơn không huỷ) + Σ khách trả, đến as_of")
    if "sem.purchase_lots" in views:
        pe = tuple(e for e in ("product",) if e in ents)
        src = ("(SELECT l.*, pr.product_name, pr.category FROM sem.purchase_lots l LEFT JOIN sem.products pr USING (product_id))"
               if "sem.products" in views else "sem.purchase_lots")
        mets["purchase_amount"] = MetricDef("purchase_amount", src, "SUM(s.quantity * s.unit_cost)", "purchase_date", pe)
        mets["purchase_qty"] = MetricDef("purchase_qty", src, "SUM(s.quantity)", "purchase_date", pe)
    for name, m in mets.items():
        if not m.description:
            m.description = METRIC_DESC.get(name, "")
        if name in MONEY_METRICS:
            m.money_decimals = rs.numbers.money_decimals
    dims = default_dimensions("{date}", "order_ts" if "order_ts" in cols else None)
    if "hour" in dims and rs.time.shifts:
        # rulebook shifts ("ca sáng 06:00–13:59") as a named dimension: filter {"field": "shift", "value": "ca sáng"}
        whens = " ".join(f"WHEN hour(s.order_ts) BETWEEN {a} AND {b} THEN '{n.replace(chr(39), '')}'"
                         for n, a, b in rs.time.shifts)
        dims["shift"] = DimensionDef("shift", f"(CASE {whens} END)", "string")
    return Catalog(mets, ents, dims, generic_sources=dict(generic_tables or {}),
                   weekday_names=weekday_names or dict(DEFAULT_WEEKDAYS))


def catalog_to_json(cat: Catalog) -> dict:
    return {"metrics": {k: asdict(v) for k, v in cat.metrics.items()},
            "entities": {k: asdict(v) for k, v in cat.entities.items()},
            "dimensions": {k: asdict(v) for k, v in cat.dimensions.items()},
            "generic_sources": cat.generic_sources,
            "weekday_names": {str(k): v for k, v in cat.weekday_names.items()}}


def catalog_from_json(d: dict) -> Catalog:
    mets = {k: MetricDef(**{**v, "entities": tuple(v["entities"])}) for k, v in d["metrics"].items()}
    ents = {k: EntityDef(**v) for k, v in d["entities"].items()}
    dims = {k: DimensionDef(**v) for k, v in d["dimensions"].items()}
    return Catalog(mets, ents, dims, d.get("generic_sources", {}), {int(k): v for k, v in d["weekday_names"].items()})


def load_weekday_names(path: str | None) -> dict[int, str]:
    try:
        with open(path, encoding="utf-8") as f:
            return {int(k): v for k, v in json.load(f).items()}
    except (OSError, TypeError, ValueError):
        return dict(DEFAULT_WEEKDAYS)
