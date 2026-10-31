"""Heuristic BSM proposal (05.2 §4.1): synonym + dtype + value + relation evidence ->
column roles; signature roles -> table pack entity; keyword status maps; gaps.
The proposal is a complete, usable BSM on its own (LLM refinement is optional)."""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from dma.ingest.loader import q
from dma.semantic.bsm import BSM, ColumnRole, LogicalTable, Relationship, dtype_compatible, load_ontology
from dma.semantic.readme import parse_readme, readme_roles
from dma.textnorm import nfc, fold

MIN_SCORE = 2.5
PACK_ORDER = ["sales_lines", "orders", "purchase_lots", "inventory", "receivables", "unit_conversions", "payments",
              "products", "customers",
              "suppliers", "staff", "branches"]
MASTER_PACKS = {"products", "customers", "suppliers", "staff", "branches"}

def _status_keywords() -> list[tuple[str, str]]:
    """Ordered (label, folded regex) from assets/ontology.yaml `status_vocabulary`."""
    return [(k, v) for k, v in (load_ontology().get("status_vocabulary") or {}).items()]


LEDGER_KEYWORDS = [("credit", r"\b(thu no|tra no|thanh toan|da tra|credit|tra|thu)\b"),
                   ("debit", r"\b(ghi no|no|mua no|phat sinh no|debit|cho no|ban chiu|chiu)\b")]
DOMAINS = [
    ("vat_lieu_xay_dung", r"xi mang|thep|gach|cat vang|son|ton lop|da 1x2|ong nhua"),
    ("ca_phe_tra_sua", r"ca phe|tra sua|bac xiu|tra dao|cappuccino|latte|sinh to|topping"),
    ("nha_thuoc", r"thuoc|vien|siro|paracetamol|vitamin|khau trang|bang gac|hop \d+ vi"),
    ("me_va_be", r"bim|ta |sua bot|binh sua|do so sinh|khan sua"),
    ("sua_xe", r"thay nhot|bugi|lop xe|ma phanh|bao duong|rua xe|nhot"),
    ("nha_hang", r"\blau\b|com |canh|nuong|goi |pho |bun |mon an"),
    ("tap_hoa", r"mi tom|nuoc ngot|bia|dau an|nuoc mam|bot giat|keo|banh"),
]


def _camel_split(s: str) -> str:
    """'MaHoaDon' -> 'Ma Hoa Don', 'SLQuyDoi' -> 'SL Quy Doi' (SQLite exports often use CamelCase)."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", s)


# abbreviations of Vietnamese POS headers ("SL bán", "Mã KH"); ambiguous ones stay out: "ck" (chuyển khoản /
# chiết khấu), "tt" (thanh toán / thành tiền) — their phrases are ontology synonyms instead
HEADER_ABBREV = {"sl": "so luong", "kh": "khach hang", "nv": "nhan vien", "ncc": "nha cung cap",
                 "dvt": "don vi tinh", "sp": "san pham", "hd": "hoa don", "gt": "gia tri", "dg": "don gia"}


def _expand_header(n: str) -> str:
    toks = n.split()
    return " ".join(HEADER_ABBREV.get(t, t) for t in toks) if len(toks) > 1 else n


def _syn_score(col: dict, syns: list[str]) -> float:
    names = {fold(col["orig_name"]), col["name"].replace("_", " "), fold(_camel_split(col["orig_name"]))}
    # unit suffixes are not part of the name: "Thành tiền (đ)", "Giảm (VND)", "Giá (nghìn)"
    paren = re.search(r"\(([^)]*)\)\s*$", col["orig_name"])
    if paren and re.fullmatch(r"d|vnd|dong|k|nghin|ngan|trieu|tr|%", fold(paren.group(1))):
        names.add(fold(col["orig_name"][:paren.start()]))
    names |= {_expand_header(n) for n in names}
    best = 0.0
    for n in names:
        if not n:
            continue
        for s in syns:
            if s.startswith("@"):                # "@hãng": whole header WITH diacritics ("Hãng" is not "Hàng")
                if nfc(col["orig_name"]).strip().lower() == nfc(s[1:]):
                    return 3.0
                continue
            exact_only = s.startswith("=")       # "=hang": whole-name only ("Hàng", never "Khách hàng")
            s = s.lstrip("=")
            if n == s:
                return 3.0
            if len(s) <= 3 or exact_only:        # "ma", "id", "ten", "sl": whole-name matches only
                continue
            if re.search(rf"(^| ){re.escape(s)}( |$)", n):
                best = max(best, 2.4)
            elif len(n) >= 4 and fuzz.ratio(n, s) >= 90:
                best = max(best, 2.0)
    return best


def _values(col: dict) -> list[str]:
    return [fold(v) for v, _ in col.get("top", [])] + [fold(e) for e in col.get("examples", []) if isinstance(e, str)]


def _value_hint(col: dict, role: str) -> float:
    vals = _values(col)
    if role in ("order_status", "line_status", "appointment_status"):
        hits = sum(any(re.search(rx, v) for _, rx in _status_keywords()) for v in vals)
        return 1.5 if vals and hits >= max(1, len(vals) // 2) else (-1.0 if col.get("distinct", 99) > 30 else 0.0)
    if role == "ledger_type":
        hits = sum(any(re.search(rx, v) for _, rx in LEDGER_KEYWORDS) for v in vals)
        return 1.5 if vals and hits >= max(1, len(vals) // 2) else -1.0
    if role.endswith("_id") and col.get("id_like"):
        return 0.5
    if role in ("category", "customer_type") and col.get("distinct", 0) == 1:
        return -2.0          # one value over the whole list ("Hàng hóa" product type) groups nothing
    if role in ("category", "channel", "payment_method", "customer_type", "unit"):
        return 0.5 if 1 <= col.get("distinct", 0) <= 60 else -1.0
    if role in ("product_name", "customer_name", "service_name", "staff_name", "supplier_name"):
        # a column with one or two values over a whole list ("Cá nhân") is a type, never a name
        if col.get("distinct", 0) <= 2 and col.get("distinct_ratio", 1.0) < 0.1:
            return -4.0
        return 0.3 if col["kind"] == "string" and col.get("distinct", 0) >= 2 else 0.0
    if role == "line_discount_pct":
        return 0.5 if (col.get("max") or 1e9) <= 100 else -2.0
    return 0.0


def score_column(col: dict, role: str, spec: dict) -> float:
    syn = _syn_score(col, spec["syn"])
    if syn <= 0:
        return 0.0
    compat = dtype_compatible(spec["dtype"], col["kind"]) or (spec["dtype"] == "date" and col["kind"] == "datetime")
    if role == "order_datetime" and not col.get("has_time"):
        compat = False
    # a "before discount" amount is never the net amount nor the discount, whatever identity it satisfies
    if role in ("line_net_amount", "line_discount_amount", "line_discount_pct") and re.search(r"(^| )(truoc (giam|ck|chiet khau|km|khuyen mai)|goc)( |$)",
                                               fold(col["orig_name"])):
        return 0.0
    return syn + (2.0 if compat else -3.0) + _value_hint(col, role)


def _identity_boost(t: dict) -> dict[tuple[str, str], float]:
    b: dict[tuple[str, str], float] = {}
    for idn in t.get("identities", []):
        cols = idn["cols"]
        if idn["kind"] == "q*p=a":
            qc, pc, a = cols
            b[(qc, "quantity")] = b[(qc, "purchase_qty")] = 1.0
            b[(pc, "unit_price")] = b[(pc, "unit_cost")] = 1.0
            b[(a, "line_net_amount")] = b[(a, "line_gross_amount")] = b[(a, "purchase_amount")] = 1.0
            b[(qc, "unit_price")] = b[(pc, "unit_price")] = b.get((pc, "unit_price"), 1.0)
        elif idn["kind"] in ("q*p-d=a", "q*p*(1-d%)=a"):
            qc, pc, d, a = cols
            b[(a, "line_net_amount")] = 2.0
            b[(d, "line_discount_amount" if idn["kind"] == "q*p-d=a" else "line_discount_pct")] = 2.0
        elif idn["kind"] == "g-d=n":
            g, d, n = cols
            b[(g, "line_gross_amount")] = 1.5
            b[(d, "line_discount_amount")] = 1.5
            b[(n, "line_net_amount")] = 2.0
    return b


README_BOOST = 6.0     # a role stated by the shop's own data dictionary beats name/value heuristics


def assign_roles(t: dict, roles: list[str], ont: dict, fk_targets: dict[str, str],
                 readme: dict[tuple[str, str], str] | None = None) -> list[tuple[float, str, str]]:
    boost = _identity_boost(t)
    readme = readme or {}
    # roles of the entity the table name points to win ties ("khach_hang.TenGoi" is a customer name)
    hinted = {r for pk, rs in ont["pack_entities"].items() if pk in MASTER_PACKS and _name_hint(t, pk, ont) for r in rs}
    cands = []
    for c in t["columns"]:
        if c["kind"] == "empty":
            continue
        stated = readme.get((t["name"], c["name"]))
        if stated == "ignore":
            continue
        for r in roles:
            s = score_column(c, r, ont["roles"][r])
            if stated == r:
                compat = dtype_compatible(ont["roles"][r]["dtype"], c["kind"]) or r in ("order_status", "note")
                s = max(s, 2.0) + (README_BOOST if compat else 1.0)
            elif stated is not None and s > 0:
                s -= 2.0
            s += boost.get((c["name"], r), 0.0) if s > 0 or boost.get((c["name"], r), 0) >= 1.5 else 0.0
            tgt = fk_targets.get(c["name"])
            if tgt and r.endswith("_id") and tgt.startswith(r[:-3]):
                s += 2.0
            if s > 0 and r in hinted:
                s += 0.5
            if s >= MIN_SCORE:
                cands.append((s, r, c["name"]))
    cands.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_r, used_c, out = set(), set(), []
    for s, r, c in cands:
        if r in used_r or c in used_c:
            continue
        used_r.add(r)
        used_c.add(c)
        out.append((s, r, c))
    return out


def _name_hint(t: dict, pack: str, ont: dict) -> float:
    n = fold(t["name"].replace("_", " ")) + " " + fold(t["source"].split("@")[0].replace("_", " ").replace("/", " "))
    return 1.5 if any(re.search(rf"(^| ){re.escape(h)}( |$)", n) for h in ont["table_hints"].get(pack, [])) else 0.0


ORDER_TRAITS = {"order_status", "order_total", "customer_name", "customer_id", "debt_amount", "paid_amount",
                "line_gross_amount", "line_net_amount", "voucher_amount", "shipping_fee"}
HEADER_EVENT_ROLES = {"purchase_id", "purchase_date", "order_id", "order_date", "order_datetime", "payment_date",
                      "ledger_date", "snapshot_date", "delivery_date"}


def _date_cols(t: dict) -> list[str]:
    return [c["name"] for c in t.get("columns", []) if c.get("kind") in ("date", "datetime")]


def _pack_score(t: dict, pack: str, assigned: dict[str, str], ont: dict, stated: set | None = None) -> float | None:
    has = assigned.__contains__
    pk = set(t.get("primary_key") or [])
    hint = _name_hint(t, pack, ont)
    is_date = any(has(r) for r in ("order_date", "order_datetime"))
    amount = any(has(r) for r in ("line_net_amount", "line_gross_amount", "unit_price", "order_total"))
    prod = has("product_id") or has("product_name") or has("service_name")
    if pack == "unit_conversions":
        if not (has("conv_factor") and (has("big_unit") or has("base_unit")) and (has("product_name") or has("product_id"))):
            return None
        return 6 + hint
    if pack == "payments":
        # one row per payment of an order: the order id repeats, the date is the PAYMENT date even when the
        # column reads like an order date ("Thời gian" of a cash book)
        dated = has("payment_date") or ((is_date or _date_cols(t)) and has("order_id") and assigned["order_id"] not in pk)
        if not (has("order_id") and has("ledger_amount") and (dated or hint)) or prod or has("quantity"):
            return None
        # an invoice list (status, order total, customer, debt per order) is the order table, not a payment book
        if set(stated or ()) & ORDER_TRAITS or any(re.search(r"(^| )(trang thai|status|tinh trang)( |$)",
                                                             fold(c["orig_name"])) for c in t.get("columns", [])):
            return None
        return 4.5 + hint
    if pack in ("products", "customers") and set(stated or ()) & HEADER_EVENT_ROLES:
        return None                     # a table with order/purchase/payment dates is not a master list
    if pack == "sales_lines":
        if not (prod and (has("quantity") or has("quantity_base") or amount) and (is_date or has("order_id"))):
            return None
        if has("order_id") and assigned["order_id"] in pk and not (has("quantity") or has("quantity_base")):
            return None                  # one row per order without quantities = order header table
        return 5 + len(assigned) * 0.3 + hint + (1 if t.get("grain") == "line" else 0)
    if pack == "orders":
        if not (has("order_id") and assigned["order_id"] in pk and is_date) or (has("quantity") and prod):
            return None
        return 4 + len(assigned) * 0.3 + hint
    if pack == "products":
        if has("product_alias") and has("product_name") and not is_date:
            return 4 + len(assigned) * 0.3 + hint          # abbreviation sheet: "Ghi tắt" -> full name, kind
        key = assigned.get("product_id") or assigned.get("product_name")
        if not key or key not in pk and not (has("product_name") and has("product_id") and assigned["product_id"] in pk):
            if not (has("product_name") and hint and not is_date):
                return None
        return 3 + len(assigned) * 0.3 + hint
    if pack == "customers":
        key = assigned.get("customer_id") or assigned.get("customer_name")
        if not key or (key not in pk and not hint):
            return None
        return 3 + len(assigned) * 0.2 + hint
    if pack == "purchase_lots":
        priced = has("unit_cost") or has("purchase_amount")
        # receipt LINES of a receipt header (date on the header, joined by receipt id): a lot list without prices
        # still gives stock = received - sold
        lined = has("purchase_id") and (has("lot_id") or has("expiry_date")) and has("purchase_qty")
        if not ((priced or lined) and (has("purchase_qty") or has("purchase_amount"))
                and (has("product_id") or has("product_name")) and (has("purchase_date") or lined)):
            return None
        return 4 + hint * 2 + len(assigned) * 0.2 if hint or has("lot_id") or has("purchase_id") else None
    if pack == "inventory":
        if not (has("qty_on_hand") and (has("product_id") or has("product_name"))):
            return None
        return 3.5 + hint + len(assigned) * 0.2
    if pack == "receivables":
        if not ((has("ledger_type") and has("ledger_amount")) or (has("debit_amount") and has("credit_amount")) or
                (hint and (has("ledger_amount") or has("debit_amount")))):
            return None
        if not (has("customer_id") or has("customer_name")):
            return None
        return 4 + hint + len(assigned) * 0.2
    if pack in ("suppliers", "staff", "branches"):
        key = {"suppliers": "supplier_id", "staff": "staff_id", "branches": "branch_id"}[pack]
        name = key.replace("_id", "_name")
        if not ((has(key) and assigned[key] in pk) or (has(name) and hint)):
            return None
        return 2 + hint
    return None


def status_map(values: list[str], kind: str = "status") -> tuple[dict[str, str], list[str]]:
    rules = _status_keywords() if kind == "status" else LEDGER_KEYWORDS
    out, unmapped = {}, []
    for v in values:
        f = fold(v)
        lab = next((lab for lab, rx in rules if re.search(rx, f)), None)
        if lab is None:
            unmapped.append(v)
            lab = "other"
        out[str(v)] = lab
    return out, unmapped


def distinct_values(con, table: str, col: str, limit: int = 200) -> list[tuple[Any, int]]:
    return con.execute(f"SELECT {q(col)}, COUNT(*) FROM typed.{q(table)} WHERE {q(col)} IS NOT NULL GROUP BY 1 "
                       f"ORDER BY 2 DESC, 1 LIMIT {limit}").fetchall()


def propose_bsm(prof: dict, con, readme: str = "") -> BSM:
    ont = load_ontology()
    pe = ont["pack_entities"]
    rinfo = parse_readme(readme)
    rroles = readme_roles(rinfo, prof, ont)
    fk_target_tbl: dict[str, dict[str, str]] = {}
    for fk in prof.get("fks", []):
        fk_target_tbl.setdefault(fk["from_table"], {})[fk["from_column"]] = fk["to_table"]
    tables_out: list[LogicalTable] = []
    roles_out: list[ColumnRole] = []
    pack_of: dict[str, str | None] = {}
    assigned_of: dict[str, dict[str, str]] = {}
    scores: dict[str, float] = {}
    ranked: dict[str, list] = {}
    tmeta0 = {t["name"]: t for t in prof["tables"]}
    for t in prof["tables"]:
        if not t.get("columns"):
            continue
        opts = []
        for pack in PACK_ORDER:
            a = assign_roles(t, pe[pack], ont, {}, rroles)
            amap = {r: c for _, r, c in a}
            stated = {r for (tn, _), r in rroles.items() if tn == t["name"]}
            s = _pack_score(t, pack, amap, ont, stated)
            if s is not None:
                s += 0.8 * len(stated & set(pe[pack]))          # README evidence for this kind of table
                opts.append((s, pack, amap))
        opts.sort(key=lambda x: (-x[0], PACK_ORDER.index(x[1])))
        ranked[t["name"]] = opts
        pack_of[t["name"]] = opts[0][1] if opts else None
        assigned_of[t["name"]] = opts[0][2] if opts else {}
        scores[t["name"]] = opts[0][0] if opts else -1.0
    _resolve_duplicates(prof, pack_of, scores, ranked)
    if not any(p == "sales_lines" for p in pack_of.values()):
        # no line table: an order-grain table with a date and money becomes the sales source
        cands = [(scores[tn], tn) for tn, opts in ranked.items()
                 if any(o[1] in ("orders",) for o in opts) or pack_of.get(tn) in (None, "orders")]
        for _, tn in sorted(cands, reverse=True):
            a = next((o[2] for o in ranked[tn] if o[1] == "orders"), None) or \
                {r: c for _, r, c in assign_roles(tmeta0[tn], pe["orders"] + ["line_gross_amount", "line_net_amount"], ont, {}, rroles)}
            if (a.get("order_date") or a.get("order_datetime")) and \
                    any(r in a for r in ("order_total", "line_net_amount", "line_gross_amount", "paid_amount")):
                pack_of[tn] = "sales_lines"
                break
    # second pass: FK targets now know their pack -> boosts id roles
    fk_pack = {}
    for tn, cols in fk_target_tbl.items():
        fk_pack[tn] = {c: (pack_of.get(tt) or "") for c, tt in cols.items()}
    for t in prof["tables"]:
        p = pack_of.get(t["name"])
        roles = pe[p] if p else sorted({r for v in pe.values() for r in v})
        if p == "sales_lines" and not any(o[1] == "sales_lines" for o in ranked.get(t["name"], [])):
            roles = sorted(set(pe["sales_lines"]) | set(pe["orders"]))
        tgt = {c: {"products": "product_x", "customers": "customer_x", "suppliers": "supplier_x", "staff": "staff_x",
                   "branches": "branch_x", "orders": "order_x"}.get(pk, "") for c, pk in fk_pack.get(t["name"], {}).items()}
        a = assign_roles(t, roles, ont, tgt, rroles) if t.get("columns") else []
        if p and _pack_score(t, p, {r: c for _, r, c in a}, ont) is None:
            a = [(s, r, c) for s, r, c in assign_roles(t, pe[p], ont, {}, rroles)]
        if p == "payments" and not any(r == "payment_date" for _, r, _ in a):
            # a payment book's only date column ("Thời gian", "NgayGioThu") is the payment date
            free = [c for c in _date_cols(t) if c not in {c2 for _, _, c2 in a}]
            if len(free) == 1:
                a = [x for x in a if x[1] not in ("order_date", "order_datetime")] + [(4.0, "payment_date", free[0])]
        assigned_of[t["name"]] = {r: c for _, r, c in a}
        for s, r, c in a:
            roles_out.append(ColumnRole(table=t["name"], column=c, role=r, confidence=round(min(1.0, s / 6), 3)))
    tmeta = {t["name"]: t for t in prof["tables"]}
    for t in prof["tables"]:
        p = pack_of.get(t["name"])
        a = assigned_of.get(t["name"], {})
        dcol = a.get("order_date") or a.get("order_datetime") or a.get("purchase_date") or a.get("ledger_date") or \
            a.get("snapshot_date") or next((c["name"] for c in t["columns"] if c["sql_type"] in ("DATE", "TIMESTAMP")), None)
        grain = {"sales_lines": "order_line", "orders": "order", "products": "product", "customers": "customer",
                 "purchase_lots": "lot", "inventory": "daily", "receivables": "other:ledger"}.get(p or "", f"other:{t.get('grain', 'table')}")
        tables_out.append(LogicalTable(name=t["name"], description=f"{p or 'table'} from {t['source']}", grain=grain,
                                       primary_key=t.get("primary_key") or [], pack_entity=p, date_col=dcol))
    # unit multipliers + status maps
    for r in roles_out:
        col = next((c for c in tmeta[r.table]["columns"] if c["name"] == r.column), None)
        if col and col.get("unit_multiplier", 1.0) != 1.0:
            r.unit_multiplier = col["unit_multiplier"]
        if r.role in ("order_status", "line_status", "appointment_status", "ledger_type"):
            vals = [str(v) for v, _ in distinct_values(con, r.table, r.column)]
            m, unmapped = status_map(vals, "ledger" if r.role == "ledger_type" else "status")
            if r.role == "order_status" and len(vals) > 20:
                # free-text note column carrying the status: a note that names no status is a completed order
                m = {k: ("completed" if v == "other" else v) for k, v in m.items()}
                r.note = "status read from a free-text note column"
            r.value_map = m
    rels = [Relationship(from_table=fk["from_table"], from_column=fk["from_column"], to_table=fk["to_table"],
                         to_column=fk["to_column"], coverage=fk["coverage"])
            for fk in prof.get("fks", []) if fk["coverage"] >= 0.95 and fk.get("to_column")]
    bsm = BSM(tables=tables_out, roles=roles_out, relationships=rels,
              unit_conversions=[list(x) for x in rinfo.conversions],
              data_end=rinfo.data_end.isoformat() if rinfo.data_end else None,
              data_start=rinfo.data_start.isoformat() if rinfo.data_start else None,
              readme_roles={f"{t}.{c}": r for (t, c), r in sorted(rroles.items())},
              packs=sorted({_pack_family(p) for p in pack_of.values() if p}),
              domain_guess=guess_domain(prof), readme_facts=readme_facts(readme))
    for t in tables_out:
        if t.date_col and t.pack_entity:
            tp = tmeta[t.name]
            c = next((c for c in tp["columns"] if c["name"] == t.date_col), None)
            if c and c.get("min"):
                bsm.date_coverage[t.name] = [str(c["min"])[:10], str(c["max"])[:10]]
    bsm.gaps = compute_gaps(bsm, prof, con)
    return bsm


UNION_PACKS = {"sales_lines", "orders", "purchase_lots", "payments", "receivables", "inventory"}


def _resolve_duplicates(prof: dict, pack_of: dict[str, str | None], scores: dict[str, float],
                        ranked: dict[str, list] | None = None) -> None:
    """One table per pack. Same-schema event tables (monthly sheets) are unioned; a table that loses
    its pack falls back to its next-best pack (e.g. a stock report losing 'products' -> 'inventory')."""
    cols = {t["name"]: tuple(sorted(c["name"] for c in t["columns"])) for t in prof["tables"]}
    rows = {t["name"]: t["n_rows"] for t in prof["tables"]}
    ranked = ranked or {}
    for _ in range(4):
        changed = False
        for pack in sorted(set(p for p in pack_of.values() if p)):
            ts = sorted([t for t, p in pack_of.items() if p == pack], key=lambda t: (-scores[t], -rows[t], t))
            if len(ts) <= 1:
                continue
            keep = ts[0]
            for t in ts[1:]:
                if pack in UNION_PACKS and cols[t] == cols[keep]:
                    continue
                if pack in MASTER_PACKS and rows[t] > rows[keep] and cols[t] != cols[keep]:
                    keep, t = t, keep
                opts = [o for o in ranked.get(t, []) if o[1] != pack and
                        (o[1] in UNION_PACKS or o[1] not in pack_of.values())]
                pack_of[t] = opts[0][1] if opts else None
                if opts:
                    scores[t] = opts[0][0]
                changed = True
        if not changed:
            break


def _pack_family(p: str) -> str:
    return {"sales_lines": "commerce", "orders": "commerce", "products": "commerce", "customers": "commerce",
            "purchase_lots": "purchasing", "inventory": "inventory", "receivables": "receivables"}.get(p, p)


def guess_domain(prof: dict) -> str:
    text = " ".join(fold(v) for t in prof["tables"] for c in t["columns"] if c["kind"] == "string"
                    for v, _ in c.get("top", []) if isinstance(v, str))
    text += " " + " ".join(fold(e) for t in prof["tables"] for c in t["columns"] for e in c.get("examples", [])
                           if isinstance(e, str))
    best, n = "other", 0
    for d, rx in DOMAINS:
        k = len(re.findall(rx, text))
        if k > n:
            best, n = d, k
    return best


def readme_facts(readme: str, limit: int = 12) -> list[str]:
    out = []
    for line in (readme or "").splitlines():
        s = line.strip(" -*#\t")
        f = fold(s)
        if 8 <= len(s) <= 300 and re.search(r"\b(cot|don vi|dvt|nghin|trieu|da tru|chua tru|gom|bao gom|khong gom|"
                                             r"luu y|chu y|nghia la|tinh theo|vat|ck|chiet khau|huy|tra hang|ghi no)\b", f):
            out.append(s)
        if len(out) >= limit:
            break
    return out


def compute_gaps(bsm: BSM, prof: dict, con=None) -> list[str]:
    has = bsm.has_role
    tmeta = {t["name"]: t for t in prof["tables"]}
    gaps = []
    if not (has("unit_cost") or has("cost_price")):
        gaps.append("cost")
    if not (has("staff_id") or has("staff_name")):
        gaps.append("staff")
    branch_cols = [(r.table, r.column) for r in bsm.roles if r.role in ("branch_id", "branch_name")]
    if not branch_cols or all(_distinct(tmeta, t, c) < 2 for t, c in branch_cols):
        gaps.append("branch")
    if not has("expiry_date"):
        gaps.append("expiry")
    ts = [(r.table, r.column) for r in bsm.roles if r.role in ("order_datetime", "order_date")]
    if not (any(_col(tmeta, t, c).get("has_time") for t, c in ts) or has("order_time")):
        gaps.append("time_of_day")
    for g, roles in (("channel", ["channel"]), ("payment_method", ["payment_method"]), ("voucher", ["voucher_amount"]),
                     ("vat", ["vat_amount"]), ("shipping", ["shipping_fee"]),
                     ("supplier", ["supplier_id", "supplier_name"]), ("customer", ["customer_id", "customer_name"]),
                     ("category", ["category"])):
        if not any(has(r) for r in roles):
            gaps.append(g)
    if not (bsm.has("receivables") or has("debt_amount")):
        gaps.append("receivables")
    if not (bsm.has("inventory") or has("qty_on_hand") or bsm.has("purchase_lots")):
        gaps.append("inventory")
    if not bsm.has("sales_lines"):
        gaps.append("revenue")
    return gaps


def _col(tmeta: dict, t: str, c: str) -> dict:
    return next((x for x in tmeta.get(t, {}).get("columns", []) if x["name"] == c), {})


def _distinct(tmeta: dict, t: str, c: str) -> int:
    return int(_col(tmeta, t, c).get("distinct") or 0)
