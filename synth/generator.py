"""DataMate-Synth v1 (07 §4): synthetic businesses in the exact split layout, with messy
files, seasonal demand (lunar Tết), cancellations/returns, a rulebook, calendar.csv,
templated informal Vietnamese questions and answers from an INDEPENDENT pandas oracle.

    python -m synth.generator --out data/synth/v1 --businesses 4 --questions 40 --seed 7

The oracle shares no code with the solvers (only the lunar algorithm for Tết dates).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from dma.timeparse import lunar

CATALOGS = {
    "vat_lieu_xay_dung": {
        "Xi măng": [("Xi măng Hà Tiên PCB40 bao 50kg", "bao", 92000), ("Xi măng Insee PCB40 bao 50kg", "bao", 95000)],
        "Thép": [("Thép cuộn D6 Hòa Phát", "kg", 17500), ("Thép cây D10 Pomina", "cây", 98000), ("Thép hộp 40x80", "cây", 245000)],
        "Gạch": [("Gạch ống 4 lỗ Tuynel", "viên", 1300), ("Gạch men 60x60 Đồng Tâm", "thùng", 285000)],
        "Cát đá": [("Cát vàng xây tô", "m3", 350000), ("Đá 1x2 xanh", "m3", 420000)],
        "Sơn": [("Sơn Dulux nội thất 5L", "thùng", 890000), ("Sơn chống thấm Kova 4kg", "thùng", 610000)],
    },
    "tap_hoa": {
        "Nước ngọt": [("Sting dâu 330ml", "chai", 10000), ("Sting dâu 500ml", "chai", 13000), ("Coca Cola lon 330ml", "lon", 10000),
                      ("Pepsi chai 1.5L", "chai", 20000)],
        "Bia": [("Bia Sài Gòn Special lon", "lon", 17000), ("Bia Tiger lon 330ml", "lon", 19000), ("Thùng bia Heineken 24 lon", "thùng", 450000)],
        "Mì gói": [("Mì Hảo Hảo tôm chua cay", "gói", 4500), ("Mì Omachi sườn", "gói", 8000)],
        "Gia vị": [("Nước mắm Nam Ngư 500ml", "chai", 38000), ("Dầu ăn Neptune 1L", "chai", 52000), ("Bột ngọt Ajinomoto 400g", "gói", 32000)],
        "Bánh kẹo": [("Bánh Chocopie hộp 12", "hộp", 55000), ("Kẹo Alpenliebe gói", "gói", 22000)],
    },
    "ca_phe_tra_sua": {
        "Cà phê": [("Cà phê sữa", "ly", 25000), ("Cà phê đen", "ly", 20000), ("Bạc xỉu", "ly", 29000)],
        "Trà": [("Trà đào cam sả", "ly", 35000), ("Trà vải", "ly", 32000), ("Trà tắc", "ly", 15000)],
        "Trà sữa": [("Trà sữa trân châu", "ly", 30000), ("Trà sữa matcha", "ly", 35000)],
        "Bánh": [("Bánh mì que", "cái", 12000), ("Bánh flan", "cái", 15000)],
    },
    "tiem_hoa": {   # "new" business type (not in train per portal list)
        "Hoa bó": [("Bó hoa hồng đỏ 20 bông", "bó", 450000), ("Bó hướng dương 5 bông", "bó", 320000)],
        "Hoa lẵng": [("Lẵng hoa khai trương", "lẵng", 1200000), ("Lẵng hoa chia buồn", "lẵng", 1500000)],
        "Hoa lẻ": [("Hoa hồng Đà Lạt", "bông", 15000), ("Hoa cúc họa mi", "bó", 60000), ("Hoa ly trắng", "cành", 45000)],
        "Phụ kiện": [("Giấy gói hoa Hàn Quốc", "tờ", 8000), ("Thiệp chúc mừng", "cái", 10000)],
    },
}
NAMES = ["Chị Lan", "Anh Tú", "Cô Ba Hồng", "Chú Năm", "Anh Hùng", "Chị Mai", "Bác Tư", "Em Vy", "Anh Khoa", "Chị Thảo",
         "Nguyễn Văn Nam", "Trần Thị Hoa", "Lê Minh", "Phạm Quốc Bảo", "Cô Hạnh", "Anh Dũng", "Chị Ngọc", "Chú Sáu",
         "Anh Phát", "Chị Trinh", "Võ Thanh Tâm", "Đặng Thu Hà", "Anh Lộc", "Chị Yến"]
STATUS = {"done": "Hoàn thành", "cancel": "Đã hủy", "return": "Trả hàng"}
RULEBOOK = os.path.join(os.path.dirname(__file__), "..", "assets", "train_rulebook.md")


def no_accent(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def vn_money(x: float) -> str:
    return f"{int(round(x)):,}".replace(",", ".")


@dataclass
class Biz:
    bid: str
    kind: str
    layout: str
    products: pd.DataFrame
    customers: pd.DataFrame
    orders: pd.DataFrame
    lines: pd.DataFrame
    purchases: pd.DataFrame | None
    start: date
    end: date


def gen_business(bid: str, kind: str, layout: str, rng: random.Random, start: date, end: date) -> Biz:
    prods = []
    i = 1
    for cat, items in CATALOGS[kind].items():
        for name, unit, price in items:
            prods.append({"product_id": f"SP{i:04d}", "product_name": name, "category": cat, "unit": unit, "price": price})
            i += 1
    products = pd.DataFrame(prods)
    ncust = rng.randint(12, len(NAMES))
    customers = pd.DataFrame([{"customer_id": f"KH{j + 1:03d}", "customer_name": NAMES[j]} for j in range(ncust)])
    weights = [rng.paretovariate(1.3) for _ in prods]
    tets = {y: lunar.tet_date(y) for y in range(start.year - 1, end.year + 2)}
    orders, lines = [], []
    oid = 1
    d = start
    while d <= end:
        lam = 6.0 if kind != "ca_phe_tra_sua" else 14.0
        if d.isoweekday() >= 6:
            lam *= 1.35
        for y, t in tets.items():
            if t - timedelta(days=17) <= d <= t + timedelta(days=9):
                lam *= 1.8 if d < t else 0.5
        if d.day in (5, 10, 25):
            lam *= 1.2
        n = _poisson(rng, lam)
        for _ in range(n):
            r = rng.random()
            st = "cancel" if r < 0.03 else ("return" if r < 0.05 else "done")
            cust = customers.iloc[rng.randrange(len(customers))]["customer_id"] if rng.random() < 0.85 else None
            k = rng.choice([1, 1, 1, 2, 2, 3, 4])
            items = rng.choices(range(len(prods)), weights=weights, k=k)
            seen = set()
            o = {"order_id": f"HD{oid:06d}", "order_date": d, "customer_id": cust, "status": st,
                 "voucher": rng.choice([0, 0, 0, 0, 10000, 20000]), "shipping": rng.choice([0, 0, 15000, 30000]),
                 "hour": rng.choice(range(7, 22))}
            for it in items:
                if it in seen:
                    continue
                seen.add(it)
                p = prods[it]
                qty = rng.choice([1, 1, 2, 2, 3, 5, 10]) if p["price"] < 100000 else rng.choice([1, 1, 2, 3])
                disc = rng.choice([0, 0, 0, 0, round(p["price"] * qty * 0.05 / 1000) * 1000])
                lines.append({"order_id": o["order_id"], "product_id": p["product_id"], "quantity": qty,
                              "unit_price": p["price"], "discount": disc, "amount": p["price"] * qty - disc})
            orders.append(o)
            oid += 1
        d += timedelta(days=1)
    purchases = None
    if kind in ("vat_lieu_xay_dung", "tiem_hoa"):
        pur = []
        lot = 1
        for p in prods:
            dd = start - timedelta(days=10)
            while dd <= end:
                pur.append({"lot_id": f"LO{lot:05d}", "product_id": p["product_id"], "purchase_date": dd,
                            "qty": rng.choice([50, 100, 200]), "unit_cost": round(p["price"] * rng.uniform(0.62, 0.8) / 100) * 100})
                lot += 1
                dd += timedelta(days=rng.randint(25, 60))
        purchases = pd.DataFrame(pur)
    return Biz(bid, kind, layout, products, customers, pd.DataFrame(orders), pd.DataFrame(lines), purchases, start, end)


def _poisson(rng: random.Random, lam: float) -> int:
    L, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= L:
            return k
        k += 1


# ---------------------------------------------------------------- writers
def write_business(b: Biz, root: str, rng: random.Random) -> None:
    d = os.path.join(root, "data", b.bid)
    os.makedirs(d, exist_ok=True)
    pmap = b.products.set_index("product_id")
    if b.layout == "xlsx":
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "DonHang"
        ws.append(["SỔ BÁN HÀNG"])
        ws.append(["Kỳ: " + b.start.strftime("%m/%Y") + " - " + b.end.strftime("%m/%Y")])
        ws.append([])
        ws.append(["Mã HĐ", "Ngày", "Mã KH", "Trạng thái", "Voucher", "Phí ship"])
        for o in b.orders.itertuples():
            ws.append([o.order_id, o.order_date.strftime("%d/%m/%Y"), o.customer_id or "", STATUS[o.status],
                       o.voucher, o.shipping])
        ws2 = wb.create_sheet("ChiTiet")
        ws2.append(["CHI TIẾT HÓA ĐƠN", None, None, None, None, None])
        ws2.append(["Mã HĐ", "Mã SP", "SL", "Đơn giá", "CK", "Thành tiền"])
        for ln in b.lines.itertuples():
            ws2.append([ln.order_id, ln.product_id, ln.quantity, vn_money(ln.unit_price), ln.discount, vn_money(ln.amount)])
        ws2.append(["Tổng cộng", None, int(b.lines.quantity.sum()), None, None, vn_money(b.lines.amount.sum())])
        wb.save(os.path.join(d, "ban_hang.xlsx"))
        with open(os.path.join(d, "san_pham.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["Mã SP", "Tên sản phẩm", "Nhóm hàng", "ĐVT", "Giá bán"])
            for p in b.products.itertuples():
                w.writerow([p.product_id, p.product_name, p.category, p.unit, vn_money(p.price)])
        with open(os.path.join(d, "khach_hang.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Mã KH", "Tên khách hàng"])
            for c in b.customers.itertuples():
                w.writerow([c.customer_id, c.customer_name])
        readme = (f"# Cửa hàng {b.kind.replace('_', ' ')}\n\nFile ban_hang.xlsx: sheet DonHang (đơn), ChiTiet (dòng hàng). "
                  "Cột Thành tiền đã trừ chiết khấu (CK). Đơn giá tính bằng đồng.\n")
    elif b.layout == "sqlite":
        con = sqlite3.connect(os.path.join(d, "shop.sqlite"))
        con.execute("CREATE TABLE products(code TEXT PRIMARY KEY, name TEXT, category TEXT, unit TEXT, list_price INTEGER)")
        con.execute("CREATE TABLE customers(customer_code TEXT PRIMARY KEY, full_name TEXT)")
        con.execute("CREATE TABLE orders(order_no TEXT PRIMARY KEY, created_at TEXT, customer_code TEXT REFERENCES customers(customer_code), "
                    "status TEXT, voucher INTEGER, shipping_fee INTEGER)")
        con.execute("CREATE TABLE order_items(order_no TEXT REFERENCES orders(order_no), code TEXT REFERENCES products(code), "
                    "qty INTEGER, price INTEGER, discount INTEGER, line_total INTEGER)")
        con.executemany("INSERT INTO products VALUES (?,?,?,?,?)",
                        [(p.product_id, p.product_name, p.category, p.unit, int(p.price)) for p in b.products.itertuples()])
        con.executemany("INSERT INTO customers VALUES (?,?)", [(c.customer_id, c.customer_name) for c in b.customers.itertuples()])
        st = {"done": "completed", "cancel": "cancelled", "return": "returned"}
        con.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)",
                        [(o.order_id, f"{o.order_date.isoformat()} {o.hour:02d}:{rng.randrange(60):02d}:00", o.customer_id,
                          st[o.status], int(o.voucher), int(o.shipping)) for o in b.orders.itertuples()])
        con.executemany("INSERT INTO order_items VALUES (?,?,?,?,?,?)",
                        [(ln.order_id, ln.product_id, int(ln.quantity), int(ln.unit_price), int(ln.discount), int(ln.amount))
                         for ln in b.lines.itertuples()])
        if b.purchases is not None:
            con.execute("CREATE TABLE purchase_lots(lot_no TEXT, code TEXT, import_date TEXT, import_qty INTEGER, unit_cost INTEGER)")
            con.executemany("INSERT INTO purchase_lots VALUES (?,?,?,?,?)",
                            [(x.lot_id, x.product_id, x.purchase_date.isoformat(), int(x.qty), int(x.unit_cost))
                             for x in b.purchases.itertuples()])
        con.commit()
        con.close()
        readme = "Dữ liệu xuất từ phần mềm bán hàng (SQLite)."
    else:   # flat csv, names only, no product codes
        rows = b.lines.merge(b.orders, on="order_id")
        cmap = b.customers.set_index("customer_id")["customer_name"].to_dict()
        with open(os.path.join(d, "giao_dich.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Ngày bán", "Số HĐ", "Tên hàng", "Loại", "Số lượng", "Giá", "Giảm giá", "Thành tiền", "Khách", "Tình trạng"])
            for r in rows.itertuples():
                w.writerow([r.order_date.strftime("%d/%m/%Y"), r.order_id, pmap.loc[r.product_id, "product_name"],
                            pmap.loc[r.product_id, "category"], r.quantity, int(r.unit_price), int(r.discount), int(r.amount),
                            cmap.get(r.customer_id, ""), STATUS[r.status]])
        readme = ""
    if b.purchases is not None and b.layout != "sqlite":
        with open(os.path.join(d, "nhap_hang.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            if b.layout == "flat":    # sales carry names only -> purchases must name the goods too
                w.writerow(["Số lô", "Tên hàng", "Ngày nhập", "SL nhập", "Giá nhập"])
                for x in b.purchases.itertuples():
                    w.writerow([x.lot_id, pmap.loc[x.product_id, "product_name"], x.purchase_date.strftime("%d/%m/%Y"),
                                x.qty, x.unit_cost])
            else:
                w.writerow(["Số lô", "Mã SP", "Ngày nhập", "SL nhập", "Giá nhập"])
                for x in b.purchases.itertuples():
                    w.writerow([x.lot_id, x.product_id, x.purchase_date.strftime("%d/%m/%Y"), x.qty, x.unit_cost])
    with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def write_calendar(path: str, start: date, end: date) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ngay_duong", "ngay_am", "thang_am", "nam_am", "thang_nhuan", "su_kien"])
        d = start
        while d <= end:
            ld = lunar.solar_to_lunar(d)
            ev = "Tết Nguyên Đán" if (ld.month, ld.day) == (1, 1) else ("Trung thu" if (ld.month, ld.day) == (8, 15) else "")
            w.writerow([d.isoformat(), ld.day, ld.month, ld.year, int(ld.leap), ev])
            d += timedelta(days=1)


# ---------------------------------------------------------------- oracle
WD = {1: "Thứ Hai", 2: "Thứ Ba", 3: "Thứ Tư", 4: "Thứ Năm", 5: "Thứ Sáu", 6: "Thứ Bảy", 7: "Chủ nhật"}


class Oracle:
    """Independent implementation of the default RuleSpec on the clean frames."""

    def __init__(self, b: Biz):
        self.b = b
        L = b.lines.merge(b.orders, on="order_id").merge(b.products[["product_id", "product_name", "category"]], on="product_id")
        self.all = L
        bad = set(b.orders[b.orders.status.isin(["cancel", "return"])].order_id)
        self.valid = L[~L.order_id.isin(bad)].copy()

    def win(self, a: date, z: date, valid: bool = True) -> pd.DataFrame:
        df = self.valid if valid else self.all
        return df[(df.order_date >= a) & (df.order_date <= z)]

    def revenue(self, a, z):
        return float(self.win(a, z).amount.sum())

    def rank(self, a, z, by="product_id", metric="quantity", k=1, asc=False):
        w = self.win(a, z)
        if w.empty:
            return None
        g = w.groupby(by).order_id.nunique() if metric == "orders" else w.groupby(by)[metric].sum()
        g = g.rename("m").reset_index()
        g = g.sort_values(["m", by], ascending=[asc, True]).reset_index(drop=True)
        if len(g) > k and g.iloc[k - 1].m == g.iloc[k].m:
            return None
        return g.head(k)

    def cogs(self, a, z, as_of):
        p = self.b.purchases
        if p is None:
            return None
        p = p[p.purchase_date <= as_of]
        avg = (p.qty * p.unit_cost).groupby(p.product_id).sum() / p.qty.groupby(p.product_id).sum()
        w = self.win(a, z)
        return float((w.quantity * w.product_id.map(avg)).sum())


def month_range(y: int, m: int) -> tuple[date, date]:
    s = date(y, m, 1)
    e = (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))
    return s, e


def make_questions(b: Biz, rng: random.Random, n: int) -> list[tuple[dict, dict]]:
    o = Oracle(b)
    out = []
    pname = b.products.set_index("product_id")["product_name"].to_dict()
    cname = b.customers.set_index("customer_id")["customer_name"].to_dict()
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        as_of = b.end - timedelta(days=rng.randint(0, 75))
        ws = as_of - timedelta(days=as_of.isoweekday() - 1)
        lw = (ws - timedelta(days=7), ws - timedelta(days=1))
        tm = (as_of.replace(day=1), as_of)
        py, pm = (as_of.year, as_of.month - 1) if as_of.month > 1 else (as_of.year - 1, 12)
        lm = month_range(py, pm)
        t = rng.randrange(14)
        q = a = None
        if t == 0:
            r = o.rank(*lw)
            if r is not None:
                code = r.iloc[0, 0]
                q = rng.choice(["sp bán dc nhiều nhất tuần rồi là gì", "tuan truoc mon nao ban chay nhat",
                                "Sản phẩm bán chạy nhất tuần trước?"])
                a = {"type": "string", "answer": pname[code], "alias": [code]}
        elif t == 1:
            q = rng.choice(["doanh thu tháng này bn", "thang nay ban dc bao nhieu tien", "Doanh thu tháng này là bao nhiêu?"])
            a = {"type": "number", "answer": round(o.revenue(*tm), 2)}
        elif t == 2:
            q = rng.choice(["doanh thu thang truoc", "tháng trước doanh số được bao nhiêu"])
            a = {"type": "number", "answer": round(o.revenue(*lm), 2)}
        elif t == 3:
            r = o.rank(*lm, k=3)
            if r is not None and len(r) == 3:
                q = rng.choice(["top 3 sp bán chạy tháng trước", "3 mat hang ban chay nhat thang truoc la gi"])
                a = {"type": "list", "answer": [pname[c] for c in r.iloc[:, 0]], "alias": [[c] for c in r.iloc[:, 0]]}
        elif t == 4:
            y2, m2 = (py, pm - 1) if pm > 1 else (py - 1, 12)
            cur, prev = o.revenue(*lm), o.revenue(*month_range(y2, m2))
            if prev:
                q = f"doanh thu tháng {pm} tăng hay giảm bao nhiêu % so với tháng {m2}"
                a = {"type": "percent", "answer": round((cur - prev) / prev * 100, 2)}
        elif t == 5:
            cat = rng.choice(sorted(b.products.category.unique()))
            tot = o.revenue(*lm)
            w = o.win(*lm)
            if tot:
                q = rng.choice([f"{cat.lower()} chiếm bao nhiêu % doanh thu tháng trước",
                                f"ty trong doanh thu cua {no_accent(cat).lower()} thang truoc"])
                a = {"type": "percent", "answer": round(float(w[w.category == cat].amount.sum()) / tot * 100, 2)}
        elif t == 6:
            n_c = (o.win(*lw, valid=False).status == "cancel").sum()
            q = rng.choice(["tuần rồi có đơn nào bị hủy ko", "tuan truoc co don huy khong"])
            a = {"type": "boolean", "answer": bool(n_c > 0)}
        elif t == 7:
            w = o.win(*lw)
            g = w.groupby("order_date").amount.sum().sort_values(ascending=False)
            if len(g) > 1 and g.iloc[0] != g.iloc[1]:
                q = rng.choice(["tuần rồi ngày nào bán được nhiều tiền nhất", "tuan truoc ngay nao doanh thu cao nhat"])
                a = {"type": "date", "answer": g.index[0].isoformat()}
        elif t == 8:
            w = o.win(*lm)
            g = w.groupby(w.order_date.map(lambda x: x.isoweekday())).order_id.nunique().sort_values(ascending=False)
            if len(g) > 1 and g.iloc[0] != g.iloc[1]:
                q = rng.choice(["tháng trước thứ mấy đông khách nhất", "thang truoc thu may co nhieu don nhat"])
                a = {"type": "string", "answer": WD[int(g.index[0])]}
        elif t == 9:
            n7 = o.win(as_of - timedelta(days=6), as_of).order_id.nunique()
            q = rng.choice(["7 ngày qua có bao nhiêu đơn", "7 ngay gan day ban duoc bao nhieu don"])
            a = {"type": "number", "answer": int(n7)}
        elif t == 10:
            pid = rng.choice(list(pname))
            w = o.win(*lm)
            nm = pname[pid]
            short = no_accent(nm).lower() if rng.random() < 0.6 else nm
            q = f"tháng trước bán được bao nhiêu {b.products.set_index('product_id').unit[pid]} {short}"
            a = {"type": "number", "answer": round(float(w[w.product_id == pid].quantity.sum()), 2)}
        elif t == 11:
            w = o.win(*lm)
            w = w[w.customer_id.notna()]
            g = w.groupby("customer_id").amount.sum().reset_index().sort_values(["amount", "customer_id"], ascending=[False, True])
            if len(g) > 1 and g.iloc[0].amount != g.iloc[1].amount:
                q = rng.choice(["khách nào mua nhiều tiền nhất tháng trước", "thang truoc kh nao chi nhieu nhat"])
                a = {"type": "string", "answer": cname[g.iloc[0].customer_id], "alias": [g.iloc[0].customer_id]}
        elif t == 12:
            q = rng.choice(["lợi nhuận tháng trước được bao nhiêu", "thang truoc lai bao nhieu"])
            cg = o.cogs(*lm, as_of)
            if cg is None:
                a = {"type": "unanswerable", "answer": None}
            else:
                a = {"type": "number", "answer": round(o.revenue(*lm) - cg, 2)}
        elif t == 13:
            tet = lunar.tet_date(as_of.year)
            s, e = lunar.lunar_to_solar(23, 12, as_of.year - 1), lunar.lunar_to_solar(10, 1, as_of.year)
            if e <= as_of and s >= b.start:
                q = rng.choice(["dịp Tết năm nay bán được bao nhiêu tiền", "tet nam nay doanh thu bao nhieu"])
                a = {"type": "number", "answer": round(o.revenue(s, e), 2)}
            _ = tet
        if q and a:
            qid = f"q_{b.bid}_{len(out):04d}"
            out.append(({"id": qid, "business_id": b.bid, "question": q, "as_of_date": as_of.isoformat()}, {"id": qid, **a}))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--businesses", type=int, default=4)
    ap.add_argument("--questions", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--months", type=int, default=16)
    ap.add_argument("--kinds", default="tap_hoa,vat_lieu_xay_dung,ca_phe_tra_sua,tiem_hoa")
    a = ap.parse_args(argv)
    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    end = date(2026, 3, 31)
    idx = end.year * 12 + end.month - 1 - (a.months - 1)
    start = date(idx // 12, idx % 12 + 1, 1)
    kinds = a.kinds.split(",")
    layouts = ["xlsx", "sqlite", "flat"]
    qs, ans = [], []
    for i in range(a.businesses):
        b = gen_business(f"biz_{900 + i:03d}", kinds[i % len(kinds)], layouts[i % len(layouts)], rng, start, end)
        write_business(b, a.out, rng)
        for qq, aa in make_questions(b, rng, a.questions):
            qs.append(qq)
            ans.append(aa)
    with open(os.path.join(a.out, "questions.jsonl"), "w", encoding="utf-8") as f:
        for x in qs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "answers.jsonl"), "w", encoding="utf-8") as f:
        for x in ans:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(RULEBOOK, encoding="utf-8") as f:
        rb = f.read()
    with open(os.path.join(a.out, "rulebook.md"), "w", encoding="utf-8") as f:
        f.write(rb)
    write_calendar(os.path.join(a.out, "calendar.csv"), start - timedelta(days=400), end + timedelta(days=60))
    print(f"wrote {len(qs)} questions for {a.businesses} businesses to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
