from datetime import date

import duckdb
import pytest

from dma.solvers.plan.compiler import (Catalog, CompileError, Compiler, EntityDef, MetricDef,
                                       default_dimensions, round_money)
from dma.solvers.plan.dmql import Plan

LINES = [
    # order_id, date, product_id, product_name, qty, net_amount, customer_id, valid
    ("D1", "2026-02-23", "SP0001", "Xi măng Hà Tiên PCB40 bao 50kg", 10, 900000, "KH1", True),
    ("D1", "2026-02-23", "SP0002", "Cát vàng m3", 2, 700000, "KH1", True),
    ("D2", "2026-02-25", "SP0002", "Cát vàng m3", 3, 1050000, "KH2", True),
    ("D3", "2026-02-28", "SP0003", "Thép D10", 6, 1200000, "KH2", True),
    ("D4", "2026-03-01", "SP0001", "Xi măng Hà Tiên PCB40 bao 50kg", 1, 90000, "KH3", True),
    ("D5", "2026-03-02", "SP0003", "Thép D10", 50, 9000000, "KH3", True),     # outside last week
    ("D6", "2026-02-24", "SP0003", "Thép D10", 99, 1, "KH1", False),          # cancelled -> excluded
    ("D7", "2025-03-02", "SP0001", "Xi măng Hà Tiên PCB40 bao 50kg", 4, 360000, "KH1", True),
]


@pytest.fixture()
def con():
    c = duckdb.connect()
    c.execute("CREATE SCHEMA sem")
    c.execute("""CREATE TABLE sem.sales_lines(order_id VARCHAR, order_date DATE, product_id VARCHAR,
                 product_name VARCHAR, quantity DOUBLE, net_amount DOUBLE, customer_id VARCHAR, is_valid BOOLEAN)""")
    c.executemany("INSERT INTO sem.sales_lines VALUES (?,?,?,?,?,?,?,?)", LINES)
    c.execute("CREATE VIEW sem.sales_lines_valid AS SELECT * FROM sem.sales_lines WHERE is_valid")
    c.execute("CREATE TABLE sem.products(product_id VARCHAR, product_name VARCHAR, list_price DOUBLE)")
    c.executemany("INSERT INTO sem.products VALUES (?,?,?)",
                  [("SP0001", "Xi măng Hà Tiên PCB40 bao 50kg", 90000), ("SP0002", "Cát vàng m3", 350000),
                   ("SP0003", "Thép D10", 200000), ("SP0004", "Gạch ống", 1500)])
    return c


@pytest.fixture()
def comp():
    src = "sem.sales_lines_valid"
    ent = ("product", "customer")
    cat = Catalog(
        metrics={
            "revenue": MetricDef("revenue", src, "SUM(s.net_amount)", "order_date", ent),
            "quantity": MetricDef("quantity", src, "SUM(s.quantity)", "order_date", ent),
            "order_count": MetricDef("order_count", src, "COUNT(DISTINCT s.order_id)", "order_date", ent),
        },
        entities={
            "product": EntityDef("product", "product_id", "product_name", "sem.products", "product_id",
                                 "product_name", {"list_price": "list_price"}),
            "customer": EntityDef("customer", "customer_id", "customer_id"),
        },
        dimensions=default_dimensions("order_date"),
    )
    return Compiler(cat)


LAST_WEEK = {"start": "2026-02-23", "end": "2026-03-01", "expr": "tuần rồi"}


def run(comp, con, **kw):
    return comp.run(Plan(**kw), con)


def test_rank_best_seller_last_week(comp, con):
    r = run(comp, con, answer_type="string", op="rank", measure={"metric": "quantity"},
            entity="product", time=LAST_WEEK, k=1)
    assert r.value == "Xi măng Hà Tiên PCB40 bao 50kg"         # 11 bags; cancelled order ignored


def test_tie_break_code_ascending(comp, con):
    # SP0001 and SP0002 both have 2 valid orders last week -> code ASC puts SP0001 first
    r = run(comp, con, answer_type="list", op="rank", measure={"metric": "order_count"},
            entity="product", time=LAST_WEEK, k=2)
    assert r.value == ["Xi măng Hà Tiên PCB40 bao 50kg", "Cát vàng m3"]
    assert "tie_at_boundary" not in r.flags
    r1 = run(comp, con, answer_type="string", op="rank", measure={"metric": "order_count"},
             entity="product", time=LAST_WEEK, k=1)
    assert "tie_at_boundary" in r1.flags          # such questions are excluded from the test set


def test_aggregate_compare_share(comp, con):
    a = run(comp, con, answer_type="number", op="aggregate", measure={"metric": "revenue"}, time=LAST_WEEK)
    assert a.value == 900000 + 700000 + 1050000 + 1200000 + 90000
    c = run(comp, con, answer_type="percent", op="compare", measure={"metric": "revenue"},
            time={"start": "2026-03-01", "end": "2026-03-04"},
            compare_time={"start": "2025-03-01", "end": "2025-03-04"}, compare_mode="pct_change")
    assert round(c.value, 2) == round((9090000 - 360000) / 360000 * 100, 2)
    s = run(comp, con, answer_type="percent", op="share", measure={"metric": "revenue"}, time=LAST_WEEK,
            filters=[{"field": "product", "op": "=", "value": "SP0002"}])
    assert round(s.value, 2) == round(1750000 / 3940000 * 100, 2)


def test_money_average_rounds_half_up_to_dong(comp, con):
    comp.cat.metrics["avg_order_value"] = MetricDef("avg_order_value", "sem.sales_lines_valid",
                                                    "SUM(s.net_amount) / COUNT(DISTINCT s.order_id)", "order_date",
                                                    ("customer",), additive=False, money_decimals=0)
    a = run(comp, con, answer_type="number", op="aggregate", measure={"metric": "avg_order_value"},
            time={"start": "2026-02-23", "end": "2026-02-28"})
    assert a.value == 1283333 and isinstance(a.value, int)          # 3 850 000 / 3 orders
    assert round_money(2.5, 0) == 3 and round_money(171266.5, 0) == 171267 and round_money(1.005, 2) == 1.01
    assert round_money(None, 0) is None and round_money(7.25, None) == 7.25


def test_argmax_time_weekday_and_day(comp, con):
    d = run(comp, con, answer_type="date", op="argmax_time", measure={"metric": "revenue"}, entity="day",
            time=LAST_WEEK)
    assert d.value == "2026-02-23"
    w = run(comp, con, answer_type="string", op="argmax_time", measure={"metric": "revenue"}, entity="weekday",
            time=LAST_WEEK)
    assert w.value == "Thứ Hai"


def test_filter_list_include_zero_count_exists_lookup(comp, con):
    z = run(comp, con, answer_type="list", op="filter_list", measure={"metric": "quantity"}, entity="product",
            time=LAST_WEEK, include_zero=True, having={"field": "measure", "op": "=", "value": 0})
    assert z.value == ["Gạch ống"]
    n = run(comp, con, answer_type="number", op="count", measure={"metric": "quantity"}, entity="customer",
            time=LAST_WEEK)
    assert n.value == 3
    e = run(comp, con, answer_type="boolean", op="exists", measure={"metric": "quantity"},
            time={"start": "2026-02-26", "end": "2026-02-27"})
    assert e.value is False
    lk = run(comp, con, answer_type="number", op="lookup", lookup_attribute="list_price",
             filters=[{"field": "product", "op": "=", "value": "SP0002"}])
    assert lk.value == 350000


def test_unknown_metric_routes_to_code_solver(comp, con):
    with pytest.raises(CompileError):
        run(comp, con, answer_type="number", op="aggregate", measure={"metric": "gross_profit"})
    with pytest.raises(CompileError):
        run(comp, con, answer_type="number", op="custom")


def test_unanswerable_plan(comp, con):
    r = run(comp, con, answer_type="unanswerable", op="aggregate", answerable=False,
            unanswerable_reason="không có dữ liệu giá vốn")
    assert r.answer_type == "unanswerable"


def test_entity_attribute_filter_and_complement(con):
    con.execute("CREATE SCHEMA meta")
    con.execute("CREATE MACRO meta.fold_vn(x) AS trim(regexp_replace(replace(strip_accents(lower(CAST(x AS VARCHAR))), "
                "'đ', 'd'), '\\s+', ' ', 'g'))")
    con.execute("ALTER TABLE sem.products ADD COLUMN material VARCHAR")
    con.execute("UPDATE sem.products SET material = 'Xi măng' WHERE product_id = 'SP0001'")
    con.execute("UPDATE sem.products SET material = 'Thép' WHERE product_id = 'SP0003'")
    src = "sem.sales_lines_valid"
    cat = Catalog(metrics={"revenue": MetricDef("revenue", src, "SUM(s.net_amount)", "order_date", ("product",))},
                  entities={"product": EntityDef("product", "product_id", "product_name", "sem.products", "product_id",
                                                 "product_name", {"material": "material"})},
                  dimensions=default_dimensions("order_date"))

    def rev(op, value):
        plan = Plan.model_validate({"answer_type": "number", "op": "aggregate", "measure": {"metric": "revenue"},
                                    "filters": [{"field": "product.material", "op": op, "value": value}],
                                    "time": LAST_WEEK})
        return Compiler(cat).run(plan, con).value

    assert rev("=", "xi mang") == 900000 + 90000          # accent/case-insensitive equality
    # complement keeps products the attribute does not cover (SP0002 has no material)
    assert rev("!=", "Thép") == 900000 + 700000 + 1050000 + 90000


def test_bare_attribute_name_resolves_to_its_entity(con):
    con.execute("CREATE SCHEMA meta")
    con.execute("CREATE MACRO meta.fold_vn(x) AS trim(regexp_replace(replace(strip_accents(lower(CAST(x AS VARCHAR))), "
                "'đ', 'd'), '\\s+', ' ', 'g'))")
    con.execute("CREATE TABLE sem.customers AS SELECT * FROM (VALUES ('KH1', 'Thân thiết'), ('KH2', 'Sỉ'), ('KH3', 'Lẻ')) "
                "t(customer_id, customer_type)")
    src = "sem.sales_lines_valid"
    cat = Catalog(metrics={"revenue": MetricDef("revenue", src, "SUM(s.net_amount)", "order_date", ("customer",))},
                  entities={"customer": EntityDef("customer", "customer_id", "customer_id", "sem.customers", "customer_id",
                                                  "customer_id", {"customer_type": "customer_type"})},
                  dimensions=default_dimensions("order_date"))
    plan = Plan.model_validate({"answer_type": "number", "op": "aggregate", "measure": {"metric": "revenue"},
                                "filters": [{"field": "customer_type", "op": "=", "value": "than thiet"}], "time": LAST_WEEK})
    assert Compiler(cat).run(plan, con).value == 900000 + 700000
    with pytest.raises(CompileError):
        Compiler(cat).run(plan.model_copy(update={"filters": [plan.filters[0].model_copy(update={"field": "nope"})]}), con)


def test_balance_date_overrides_as_of_in_balance_metrics(con):
    con.execute("CREATE TABLE sem.debt(order_id VARCHAR, order_date DATE, customer_id VARCHAR, remaining DOUBLE, pay_date DATE)")
    con.executemany("INSERT INTO sem.debt VALUES (?,?,?,?,?)",
                    [("D1", "2026-02-01", "KH1", 100, "2026-02-20"), ("D2", "2026-02-10", "KH2", 50, None)])
    src = "(SELECT * FROM sem.debt WHERE order_date <= $as_of AND (pay_date IS NULL OR pay_date > $as_of))"
    cat = Catalog(metrics={"debt_outstanding": MetricDef("debt_outstanding", src, "SUM(s.remaining)", "order_date", ("customer",))},
                  entities={"customer": EntityDef("customer", "customer_id", "customer_id")},
                  dimensions=default_dimensions("order_date"))
    comp = Compiler(cat, as_of=date(2026, 3, 1))
    assert comp.run(Plan(answer_type="number", op="aggregate", measure={"metric": "debt_outstanding"}), con).value == 50
    plan = Plan(answer_type="number", op="aggregate", measure={"metric": "debt_outstanding"}, balance_date="2026-02-15")
    assert comp.run(plan, con).value == 150
