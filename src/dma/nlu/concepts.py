"""Concepts, concept-gap mapping and type cues over the FOLDED question (05.4 §5–6)."""
from __future__ import annotations

import re

CONCEPTS: dict[str, str] = {
    "revenue": r"\b(doanh thu|doanh so|ban duoc bao nhieu tien|thu duoc|tien ban hang|ban duoc (bao nhieu|may) (tien|trieu)|tong tien ban)\b",
    "best_seller": r"\b(ban chay|ban duoc nhieu|ban nhieu nhat|hot nhat|dat hang nhat|ban dat nhat|mua nhieu nhat)\b",
    "slow_seller": r"\b(ban cham|ban e|it nguoi mua|ton nhieu|khong ban duoc|ban it nhat|e nhat)\b",
    "profit": r"\b(loi nhuan|tien lai|lai gop|lai rong|lai bao nhieu|lai duoc|bien loi|bi lo|lo bao nhieu|lo hay lai|lai hay lo|co lai|duoc lai|loi bao nhieu)\b",
    "cost": r"\b(gia von|gia nhap|chi phi nhap|tien von|von hang)\b",
    "debt": r"\b(cong no|con no|no bao nhieu|dang no|ghi no|ghi so|thieu tien|chua tra tien|khach no|no nhieu|no tien|tien no|so no)\b",
    "stock": r"\b(ton kho|con hang|het hang|hang ton|con bao nhieu (cai|hop|chai|bao|thung|goi|san pham)?|trong kho|sap het)\b",
    "expiry": r"\b(het han|hsd|han su dung|can date|sap het han|qua han|date)\b",
    "cancel": r"\b(huy|bom hang|boom|bung don|bi huy)\b",
    "return": r"\b(tra hang|tra lai|doi tra|hoan hang|bi tra)\b",
    "customer": r"\b(khach|khach hang|kh|nguoi mua)\b",
    "staff": r"\b(nhan vien|nv|thu ngan|ai ban|nguoi ban|nhan vien nao)\b",
    "branch": r"\b(chi nhanh|co so|cua hang so|chi nhanh nao)\b",
    "time_of_day": r"\b(khung gio|may gio|gio nao|buoi sang|buoi trua|buoi chieu|buoi toi|ca toi|ca sang|gio cao diem)\b",
    "growth": r"\b(tang|giam|so voi|chenh lech|tang truong|thay doi)\b",
    "share": r"\b(chiem|ty trong|ti trong|ty le|ti le|phan tram|bao nhieu %)\b",
    "average": r"\b(trung binh|binh quan|moi ngay|moi don|tb)\b",
    "forecast": r"\b(du bao|du doan|thang sau|tuan sau|nam sau|se ban|ngay mai|sap toi|tuong lai)\b",
    "opinion": r"\b(nen nhap|nen ban|tot khong|co nen|nen lam gi|loi khuyen)\b",
    "channel": r"\b(online|grab|shopee|tai quan|kenh|tiktok|lazada|ban online|giao hang tan noi|mang ve)\b",
    "payment": r"\b(tien mat|chuyen khoan|quet the|the tin dung|momo|zalopay|hinh thuc thanh toan|vi dien tu)\b",
    "voucher": r"\b(voucher|ma giam gia|coupon|khuyen mai don)\b",
    "vat": r"\b(vat|thue gtgt|tien thue)\b",
    "shipping": r"\b(phi ship|phi giao hang|tien ship|van chuyen)\b",
    "supplier": r"\b(nha cung cap|ncc|nguon hang|lay hang o dau)\b",
    "discount": r"\b(chiet khau|giam gia|ck)\b",
    "order_count": r"\b(bao nhieu don|may don|so don|bao nhieu hoa don|so hoa don|luot mua)\b",
    "quantity": r"\b(bao nhieu (cai|chai|lon|ly|bao|thung|goi|hop|kg|vien|cay|m3|phan|suat|to|dia|bo|bong|lang|canh|to|tui|chiec)|so luong|may (ly|chai|cai|thung))\b",
}
CONCEPT_GAP = {"revenue": "revenue", "profit": "cost", "cost": "cost", "debt": "receivables", "stock": "inventory",
               "expiry": "expiry", "staff": "staff", "branch": "branch", "time_of_day": "time_of_day",
               "channel": "channel", "payment": "payment_method", "voucher": "voucher", "vat": "vat",
               "shipping": "shipping", "supplier": "supplier"}
GAP_REASON_VI = {
    "cost": "Dữ liệu không có giá vốn/giá nhập nên không tính được lợi nhuận hoặc giá vốn",
    "receivables": "Dữ liệu không có thông tin công nợ",
    "inventory": "Dữ liệu không có thông tin tồn kho",
    "expiry": "Dữ liệu không có hạn sử dụng",
    "staff": "Dữ liệu không ghi nhân viên bán hàng",
    "branch": "Dữ liệu không có thông tin chi nhánh",
    "time_of_day": "Dữ liệu không có giờ bán",
    "channel": "Dữ liệu không có kênh bán",
    "payment_method": "Dữ liệu không có hình thức thanh toán",
    "voucher": "Dữ liệu không có voucher",
    "vat": "Dữ liệu không có VAT",
    "shipping": "Dữ liệu không có phí giao hàng",
    "supplier": "Dữ liệu không có nhà cung cấp",
    "revenue": "Dữ liệu không có giao dịch bán hàng",
    "forecast": "Câu hỏi về tương lai/dự báo, dữ liệu không chứa thông tin này",
    "out_of_coverage": "Kỳ được hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu của cửa hàng",
}


def detect_concepts(folded: str) -> list[str]:
    return [c for c, rx in CONCEPTS.items() if re.search(rx, folded)]


TYPE_RULES: list[tuple[str, str, float]] = [
    ("percent", r"(bao nhieu phan tram|phan tram|%|\bty le\b|\bti le\b|chiem bao nhieu|tang truong|ty trong|ti trong|"
                r"(tang|giam) (hay (tang|giam) )?bao nhieu %|bien loi nhuan)", 0.9),
    ("boolean", r"(\bco\b .* \b(khong|ko|k|hok|hong|khum)\s*\??$|phai khong|dung khong|\bco phai\b|hay khong|\blieu\b|"
                r"\bco\b .* \bchua\s*\??$|duoc khong\s*\??$)", 0.85),
    ("date", r"\b(ngay nao|hom nao|khi nao|bao gio|lan cuoi|dau tien|thang nao|ngay may)\b", 0.8),
    ("list", r"(\btop ?\d+\b|\b\d+ (mon|san pham|sp|mat hang|khach|loai|nhom|mau|dich vu)\b|\bnhung .* nao\b|\bliet ke\b|"
             r"\bke ten\b|\bcac .* nao\b|\b(ba|hai|bon|nam) (mon|san pham|mat hang|khach)\b)", 0.8),
    ("table", r"\b(kem (theo )?|cung voi|moi .* bao nhieu|theo tung|tung (mon|san pham|khach|thang|ngay)|bang)\b", 0.6),
    ("number", r"(\bbao nhieu\b|\bmay (don|ly|cai|chai|lan|ngay|khach|san pham|mon|hop|thung|goi|bao)\b|\btong\b|\btrung binh\b|\bbn\b)", 0.7),
    ("string", r"(\bthu may\b|\bla gi\b|\b(san pham|mon|khach|chi nhanh|ncc|nhan vien|mat hang|hang|loai|nhom) nao\b|\bai\b|\bthu may\b|"
               r"\bten (gi|la gi)\b)", 0.7),
]


def type_cues(folded: str) -> list[tuple[str, float]]:
    out = []
    for t, rx, w in TYPE_RULES:
        if re.search(rx, folded):
            out.append((t, w))
    if re.search(r"\bthu may\b", folded):
        out = [("string", 0.85)] + [x for x in out if x[0] != "string"]
    # "thang nao" -> date (yyyy-mm); "thu may" -> string weekday
    out.sort(key=lambda x: -x[1])
    return out


_K_NOUN = r"(mon|san pham|sp|mat hang|khach|khach hang|loai|nhom|ncc|nha cung cap|dich vu|chi nhanh|nhan vien|ngay|thang)"
_NUMW = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9, "muoi": 10}


def detect_k(folded: str) -> int | None:
    m = re.search(r"\btop ?(\d{1,3})\b", folded)
    if m:
        return int(m.group(1))
    m = re.search(rf"\b(\d{{1,2}}|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi) {_K_NOUN}\b", folded)
    if m:
        tok = m.group(1)
        noun = m.group(2)
        if noun in ("ngay", "thang"):
            return None
        if tok == "nam" and noun in ("ngay", "thang"):
            return None
        return int(tok) if tok.isdigit() else _NUMW.get(tok)
    return None


def ranking_direction(folded: str) -> str | None:
    if re.search(r"\b(it nhat|thap nhat|cham nhat|nho nhat|e nhat|kem nhat|ban it|ban cham|thap)\b", folded):
        return "asc"
    if re.search(r"\b(nhieu nhat|cao nhat|chay nhat|lon nhat|top|dong nhat|dat nhat|manh nhat|nhat)\b", folded):
        return "desc"
    return None
