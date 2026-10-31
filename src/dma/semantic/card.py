"""Business Card — prompt layer L3 (05.2 §8). Deterministic serialization (sorted, no
timestamps) so the prefix is identical for every question of the business."""
from __future__ import annotations

from dma.semantic.bsm import BSM
from dma.solvers.plan.compiler import Catalog

DOMAIN_VI = {"vat_lieu_xay_dung": "vật liệu xây dựng", "ca_phe_tra_sua": "cà phê/trà sữa", "nha_thuoc": "nhà thuốc",
             "me_va_be": "mẹ và bé", "sua_xe": "sửa xe", "nha_hang": "nhà hàng", "tap_hoa": "tạp hóa"}


def _sample(con, sql: str, k: int = 12) -> list[str]:
    try:
        return [str(r[0]) for r in con.execute(sql + f" LIMIT {k}").fetchall() if r[0] is not None]
    except Exception:  # noqa: BLE001
        return []


def render_business_card(con, bsm: BSM, cat: Catalog, views: list[str], view_cols: dict[str, list[str]],
                         prof: dict, readme: str, notes: dict[str, str], max_chars: int = 16000) -> str:
    out = ["# BUSINESS CARD  (do not use the business id for logic)"]
    out.append(f"domain: {DOMAIN_VI.get(bsm.domain_guess, bsm.domain_guess)} · packs: {', '.join(bsm.packs) or 'generic'} · currency VND")
    cov = " · ".join(f"{t} {a}..{b}" for t, (a, b) in sorted(bsm.date_coverage.items()))
    if cov:
        out.append(f"data range: {cov}")
    if bsm.readme_facts:
        out.append("README facts: " + "; ".join(f'"{x}"' for x in bsm.readme_facts[:8]))
    if notes:
        out.append("interpretation: " + "; ".join(f"{k}: {v}" for k, v in sorted(notes.items())))
    if "sem.sales_lines" in views:
        out.append("## semantic views (prefer these)")
        cols = view_cols.get("sem.sales_lines", [])
        cats = _sample(con, "SELECT category FROM sem.categories ORDER BY 1") if "sem.categories" in views else []
        stat = _sample(con, "SELECT DISTINCT order_status FROM sem.sales_lines ORDER BY 1")
        desc = []
        for c in cols:
            if c == "category" and cats:
                desc.append(f"category[{len(cats)}: {', '.join(cats[:10])}{', ...' if len(cats) > 10 else ''}]")
            elif c == "order_status":
                desc.append(f"order_status[{'|'.join(stat)}]")
            else:
                desc.append(c)
        out.append("sem.sales_lines_valid (valid orders per rulebook) and sem.sales_lines (all lines incl. cancelled/returned) "
                   "cols: " + ", ".join(desc))
        n = {v: _sample(con, f"SELECT COUNT(*) FROM {v}", 1) for v in ("sem.products", "sem.customers") if v in views}
        out.append("sem.orders (valid, one row per order) · sem.orders_all · "
                   + " · ".join(f"{v}({(n.get(v) or ['?'])[0]})" for v in ("sem.products", "sem.customers") if v in views))
    for v in ("sem.purchase_lots", "sem.inventory", "sem.receivables"):
        if v in views:
            out.append(f"{v}" + {"sem.purchase_lots": "(product_id, purchase_date, lot_id, quantity, unit_cost, expiry_date)",
                                 "sem.inventory": "(product_id, snapshot_date, qty_on_hand)",
                                 "sem.receivables": "(customer_id, ledger_date, ledger_type[debit|credit], signed_amount)"}[v])
    if "sem.product_cost" in views:
        out.append("sem.product_cost(cutoff DATE) -> (product_id, avg_unit_cost)")
    out.append("## metrics: " + ", ".join(f"{k}" for k in sorted(cat.metrics)))
    out.append("## entities: " + ", ".join(f"{k}(code={e.key})" for k, e in sorted(cat.entities.items())))
    for a in view_cols.get("sem.products", []):     # side-table attributes: filter as {"field": "product.<attr>", ...}
        vals = _sample(con, f"SELECT DISTINCT {a} FROM sem.products WHERE {a} IS NOT NULL ORDER BY 1", 8)
        out.append(f"## product attribute (filter field product.{a}; values compared ignoring case/accents; "
                   f"negation = every other product): {', '.join(vals)}")
    if "sem.customers" in views:
        vals = _sample(con, "SELECT DISTINCT customer_type FROM sem.customers WHERE customer_type IS NOT NULL ORDER BY 1", 12)
        if vals:
            out.append(f"## customer attribute (filter field customer.customer_type; values compared ignoring case/accents): "
                       f"{', '.join(vals)}")
    out.append("## dimensions: " + ", ".join(sorted(cat.dimensions)))
    out.append("## NOT AVAILABLE in the data (gaps): " + (", ".join(bsm.gaps) or "none"))
    raw = " · ".join(f"typed.{t['name']} ({t['n_rows']:,})" for t in prof["tables"])
    out.append(f"## raw tables: {raw}")
    for t in bsm.tables:
        if t.pack_entity is None:
            tp = next((x for x in prof["tables"] if x["name"] == t.name), None)
            if tp:
                out.append(f"typed.{t.name} cols: " + ", ".join(f"{c['name']}:{c['kind']}" for c in tp["columns"][:30]))
    if readme and readme.strip():
        out.append("## SHOP README (verbatim — authoritative for how THIS shop's files must be read)")
        out.append(readme.strip()[:5000])
    warns = [w for w in bsm.warnings][:6]
    if warns:
        out.append("warnings: " + "; ".join(warns))
    card = "\n".join(out)
    return card if len(card) <= max_chars else card[:max_chars] + "\n…"
