"""Deterministic rule-item -> RuleSpec patch extraction (accent-insensitive regexes).

Used (1) as a strong voter next to the LLM samples for changed items, (2) alone when the
LLM is off, (3) to bootstrap train_rulespec.json from a rulebook (then hand-verified).
Only phrases that EXPLICITLY state a value produce a patch; everything else is unmapped.
"""
from __future__ import annotations

import re
from typing import Any

from dma.textnorm import fold, strip_accents

_LUNAR_MONTH = {"chap": 12, "giap": 12, "mot": 11, "gieng": 1, "gieng am": 1}


def _f(text: str) -> str:
    for a, b in ((">=", " it nhat "), ("≥", " it nhat "), ("<=", " toi da "), ("≤", " toi da "),
                 (">", " tren "), ("<", " duoi "), ("%", " phan tram ")):
        text = text.replace(a, b)
    return fold(text)


def _num(s: str) -> int | None:
    words = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7, "tam": 8, "chin": 9,
             "muoi": 10}
    return int(s) if s.isdigit() else words.get(s)


def extract_patch(text: str) -> tuple[dict[str, Any], list[str]]:
    """-> (patch {field.path: value}, touched_fields). touched without a value = field mentioned
    but not decidable (the LLM / verbatim rules handle it)."""
    t = _f(text)
    p: dict[str, Any] = {}
    touched: list[str] = []

    def touch(*fields):
        touched.extend(f for f in fields if f not in touched)

    # ---------------------------------------------------------------- revenue
    if re.search(r"^\W*doanh (thu|so)\b", t) or re.search(r"\bdoanh (thu|so) (la|duoc tinh|duoc hieu|tinh bang|bang)\b", t):
        touch("revenue.base", "revenue.subtract_order_voucher", "revenue.include_vat", "revenue.include_shipping")
        if re.search(r"(sau|da tru) (chiet khau|ck|giam gia)( cap)?( dong| tung dong| mat hang)?", t):
            p["revenue.base"] = "line_net"
        elif re.search(r"truoc (chiet khau|ck|giam gia)", t):
            p["revenue.base"] = "line_gross"
        elif re.search(r"so luong (x|nhan|\*) (voi )?don gia", t):
            p["revenue.base"] = "qty_x_price"
        if re.search(r"khong (tru|tinh tru|bi tru)( di)? (voucher|ma giam gia|khuyen mai cap don|giam gia cap don)", t):
            p["revenue.subtract_order_voucher"] = False
        elif re.search(r"(tru|tru di|tru ca|da tru|sau khi tru) (ca )?(voucher|ma giam gia)", t):
            p["revenue.subtract_order_voucher"] = True
        if re.search(r"(khong|chua) (gom|bao gom|tinh|co|cong)( ca)? (thue )?(vat|thue)", t):
            p["revenue.include_vat"] = False
        elif re.search(r"(gom|bao gom|tinh ca|da gom|cong ca|cong) (ca )?(thue )?(vat|thue gtgt)", t):
            p["revenue.include_vat"] = True
        ship = r"(phi giao hang|phi ship|phi van chuyen|tien ship|ship)"
        if re.search(rf"(khong|chua) (gom|bao gom|tinh|cong)( ca)? {ship}", t) or \
                re.search(rf"(khong|chua) (gom|bao gom|tinh)( ca)? (thue )?(vat|thue)( va| hay| hoac|,) {ship}", t):
            p["revenue.include_shipping"] = False
        elif re.search(rf"(gom|bao gom|tinh ca|cong ca|cong) (ca )?{ship}", t):
            p["revenue.include_shipping"] = True

    # ------------------------------------------------------------ valid order
    about_debt = bool(re.search(r"\b(cong no|tinh vao no|con no|ghi no|tuoi no)\b", t))
    if about_debt:
        pass                               # "đơn được tính vào nợ" is a debt rule, not the revenue valid-order rule
    elif re.search(r"^\W*(don (duoc tinh|hop le)|don hang (duoc tinh|hop le)|don tinh vao doanh thu)\b", t) or \
            (re.search(r"\b(huy|tra lai|tra hang|bi tra)\b", t) and re.search(r"\b(loai|khong tinh)\b", t)):
        touch("valid_order.exclude_order_if_any_line_returned", "valid_order.exclude_returned_lines_only",
              "valid_order.canceled_statuses")
        if re.search(r"\b(bo|loai) (ca|toan bo) don\b", t) and re.search(r"tra", t) and not re.search(r"khong (bo|loai) (ca|toan bo) don", t) \
                and not re.search(r"chi (loai|bo) (cac |nhung )?(dong|mat hang|san pham)", t):
            p["valid_order.exclude_order_if_any_line_returned"] = True
            p["valid_order.exclude_returned_lines_only"] = False
        elif re.search(r"loai (ca|toan bo) don (co|chua) (bat ky|it nhat mot|mot|bat cu) (mat hang|dong|san pham|mon)", t) \
                or re.search(r"don co (bat ky|bat cu) (mat hang|dong|san pham)[a-z ]* (bi )?tra", t):
            p["valid_order.exclude_order_if_any_line_returned"] = True
            p["valid_order.exclude_returned_lines_only"] = False
        elif re.search(r"chi (loai|bo) (cac |nhung )?(dong|mat hang|san pham)[a-z ]*(bi )?tra", t) or \
                re.search(r"khong (loai|bo) (ca|toan bo) don", t):
            p["valid_order.exclude_order_if_any_line_returned"] = False
            p["valid_order.exclude_returned_lines_only"] = True
        elif re.search(r"(tra hang|bi tra|tra lai)[a-z ]*(van|cung) (duoc )?tinh", t):
            p["valid_order.exclude_order_if_any_line_returned"] = False
            p["valid_order.exclude_returned_lines_only"] = False
        if re.search(r"\bhuy\b", t) and re.search(r"\b(loai|khong tinh)\b", t):
            p["valid_order.canceled_statuses"] = ["canceled"]

    # ----------------------------------------------------------------- ranking
    if re.search(r"\b(ban chay|ban duoc nhieu)\b", t):
        touch("ranking.best_seller_metric")
        if re.search(r"theo (tong )?(so luong|sl)\b", t):
            p["ranking.best_seller_metric"] = "quantity"
        elif re.search(r"theo (tong )?(doanh thu|doanh so|thanh tien|so tien)\b", t):
            p["ranking.best_seller_metric"] = "revenue"
        elif re.search(r"theo (tong )?so (don|luot mua|hoa don)\b", t):
            p["ranking.best_seller_metric"] = "order_count"
    if re.search(r"\bban cham\b", t) and re.search(r"\b(la|duoc hieu|nghia la|=|:)\b", t):
        touch("ranking.slow_seller_definition", "ranking.include_zero_activity")
        p["ranking.slow_seller_definition"] = text.strip()
        if re.search(r"(ke ca|bao gom|tinh ca) (san pham|mat hang)[a-z ]*(khong ban|chua ban|0)", t):
            p["ranking.include_zero_activity"] = True
        elif re.search(r"(chi xet|chi tinh) (san pham|mat hang)[a-z ]*(co ban|phat sinh|da ban)", t):
            p["ranking.include_zero_activity"] = False
    if re.search(r"\b(hoa|bang nhau|dong hang)\b", t) and re.search(r"\bma\b[a-z ]*tang dan", t):
        touch("ranking.tie_break")
        p["ranking.tie_break"] = "code_asc"
    elif re.search(r"\b(hoa|bang nhau)\b", t) and re.search(r"\bten\b[a-z ]*(tang dan|a z|abc)", t):
        touch("ranking.tie_break")
        p["ranking.tie_break"] = "name_asc"

    # -------------------------------------------------------------------- time
    if re.search(r"\b(mot|1) tuan\b|\btuan (duoc )?(tinh|bat dau)\b", t):
        touch("time.week_start")
        if re.search(r"bat dau (tu |vao |la |bang )?(chu nhat|cn)\b", t):
            p["time.week_start"] = "sunday"
        elif re.search(r"bat dau (tu |vao |la |bang )?(thu hai|thu 2)\b", t):
            p["time.week_start"] = "monday"
        elif re.search(r"(tu |bat dau (tu |vao )?)?(thu hai|thu 2)( den| toi| -|\.\.)? (het )?(chu nhat|cn)", t):
            p["time.week_start"] = "monday"
        elif re.search(r"(tu |bat dau (tu |vao )?)?(chu nhat|cn)( den| toi| -|\.\.)? (het )?(thu bay|thu 7)", t):
            p["time.week_start"] = "sunday"
    if re.search(r"\b(thang nay|tuan nay|nam nay|quy nay)\b", t):
        touch("time.current_period_to_date")
        if re.search(r"(den|toi) (het )?(chinh )?(ngay do|ngay hom nay|hom nay|as of|ngay as of|ngay hien tai)", t):
            p["time.current_period_to_date"] = True
        elif re.search(r"(ca|tron|toan bo) (thang|tuan|nam|quy)\b", t):
            p["time.current_period_to_date"] = False
    if re.search(r"\b(n|\d+) ngay (qua|gan day|gan nhat)\b", t):
        touch("time.last_n_days_includes_today")
        if re.search(r"(khong|chua) (tinh|gom|bao gom|ke)( ca)? (hom nay|ngay as of|ngay hien tai)", t):
            p["time.last_n_days_includes_today"] = False
        elif re.search(r"(tinh|gom|bao gom|ke) (ca )?(hom nay|ngay as of|ngay hien tai)", t):
            p["time.last_n_days_includes_today"] = True
    if re.search(r"\btet\b", t):
        a = re.search(r"\btu\b.*?\b(\d{1,2}) (?:thang )?(chap|giap)\b", t)
        b = re.search(r"\b(?:den|toi)\b.*?\b(?:mung|mong) (\d{1,2})(?: (?:thang )?(gieng|\d{1,2}))?", t)
        if a and b:
            touch("time.tet_start_lunar", "time.tet_end_lunar")
            p["time.tet_start_lunar"] = (12, int(a.group(1)))
            em = b.group(2)
            p["time.tet_end_lunar"] = (1 if em in (None, "gieng") else int(em), int(b.group(1)))
    if re.search(r"\btrung thu\b", t):
        touch("time.mid_autumn_window")
        b = re.search(r"truoc (?:do )?(\d+|mot|hai|ba|bon|nam|sau|bay) ngay", t)
        a = re.search(r"sau (?:do )?(\d+|mot|hai|ba|bon|nam|sau|bay) ngay", t)
        if b or a:
            p["time.mid_autumn_window"] = (_num(b.group(1)) if b else 0, _num(a.group(1)) if a else 0)
        elif re.search(r"(chi|dung) (ngay )?(ram|15) thang (tam|8)", t):
            p["time.mid_autumn_window"] = (0, 0)
    if re.search(r"\bcuoi tuan\b", t):
        touch("time.weekend_isodays")
        if re.search(r"thu (bay|7) (va|,) chu nhat", t):
            p["time.weekend_isodays"] = [6, 7]
        elif re.search(r"chi (ngay )?chu nhat", t):
            p["time.weekend_isodays"] = [7]
        elif re.search(r"thu (sau|6),? thu (bay|7) (va|,) chu nhat", t):
            p["time.weekend_isodays"] = [5, 6, 7]
    if re.search(r"\bngay (nhan|lanh|tra|phat) luong\b", t):
        touch("time.payday_days")
        days = [int(x) for x in re.findall(r"\bngay (\d{1,2})\b", t) if 1 <= int(x) <= 31]
        if days:
            p["time.payday_days"] = sorted(set(days))
    if re.search(r"\bthang \d+\b[a-z ]*khong (co |ghi )?nam\b|\bthang (khong|ko) (kem|ghi) nam\b", t):
        touch("time.bare_month_policy")
        if re.search(r"(nam hien tai|nam nay|cung nam)", t):
            p["time.bare_month_policy"] = "current_year"
        elif re.search(r"(gan nhat|truoc do|da qua)", t):
            p["time.bare_month_policy"] = "latest_not_after_as_of"

    if re.search(r"\bthang nay\b", t) and re.search(r"thang truoc", t):
        touch("time.mtd_compare_same_days")
        if re.search(r"cung so ngay|den het ngay trung so", t):
            p["time.mtd_compare_same_days"] = True
        elif re.search(r"(ca|tron|toan bo) thang truoc", t):
            p["time.mtd_compare_same_days"] = False
    plain = re.sub(r"\s+", " ", strip_accents(text.lower()).replace("đ", "d").replace("–", "-"))   # keeps "06:00"
    sh = re.findall(r"\b(ca [a-z]+) (\d{1,2})[:h]\d{2} ?(?:-|den|toi)? ?(\d{1,2})[:h](\d{2})", plain)
    if sh:
        touch("time.shifts")
        orig = {fold(m): m for m in re.findall(r"\b[Cc]a [^\s,.:;0-9*_]+", text)}
        p["time.shifts"] = [(orig.get(n, n), int(a), int(b)) for n, a, b, _ in sh]
    # -------------------------------------------------------------------- cost
    if re.search(r"\bgia von\b", t):
        touch("cost.method", "cost.lots_cutoff")
        if re.search(r"binh quan gia quyen|trung binh gia quyen", t):
            p["cost.method"] = "weighted_avg_by_lot"
        elif re.search(r"\bfifo\b|nhap truoc xuat truoc", t):
            p["cost.method"] = "fifo"
        elif re.search(r"(gia nhap|lan nhap|lo nhap) (gan nhat|moi nhat|cuoi cung)", t):
            p["cost.method"] = "latest_cost"
        elif re.search(r"cot gia von|gia von (co dinh|niem yet)", t):
            p["cost.method"] = "static_cost_price"
        if re.search(r"gia von (ghi )?(tren|o|cua) (moi )?dong ban", t):
            touch("cost.profit_source")
            p["cost.profit_source"] = "sale_line"
        if re.search(r"(cac lo|lo) nhap (truoc|den|toi) (hoac vao )?ngay ban|theo lo nhap,? tai (thoi diem|luc|ngay) ban", t):
            p["cost.lots_cutoff"] = "until_sale_date"
        elif re.search(r"(tat ca|moi|toan bo) (cac )?lo( nhap)?", t):
            p["cost.lots_cutoff"] = "all"
        elif re.search(r"(den|toi) (ngay )?(as of|hom nay|ngay hien tai)", t):
            p["cost.lots_cutoff"] = "until_as_of"

    # ------------------------------------------------------------------ numbers
    m = re.search(r"lam tron (?:den |toi )?(\d|hai|mot|ba) (chu so|so) (thap phan|le)", t)
    if m:
        touch("numbers.decimals")
        p["numbers.decimals"] = _num(m.group(1))
    return p, touched


def extract_vocab(text: str) -> dict | None:
    """'Khách quen: khách có ≥ 5 đơn hợp lệ' / '"Đơn lớn" là đơn trên 2 triệu' -> VocabEntry dict."""
    s = text.strip()
    m = re.match(r'^\s*[*_"“]*([^:"”*_=]{2,40}?)[*_"”]*\s*(?::|=|\blà\b|\bla\b|được hiểu là|nghĩa là)\s*(.+)$', s)
    if not m:
        return None
    term, definition = m.group(1).strip(" -"), m.group(2).strip()
    if len(term.split()) > 6 or len(definition) < 5:
        return None
    t = _f(definition)
    params: dict[str, Any] = {}
    mm = re.search(r"(tu|it nhat|toi thieu|tren) (\d+) (don|lan|luot|hoa don)", t)
    if mm:
        params["min_orders"] = int(mm.group(2)) + (1 if mm.group(1) == "tren" else 0)
    mm = re.search(r"(\d+) ngay", t)
    if mm:
        params["days"] = int(mm.group(1))
    mm = re.search(r"(tren|tu|it nhat) ([\d.,]+) ?(trieu|tr|nghin|k|dong)?", t)
    if mm and "min_orders" not in params:
        try:
            v = float(mm.group(2).replace(".", "").replace(",", "."))
            v *= {"trieu": 1e6, "tr": 1e6, "nghin": 1e3, "k": 1e3}.get(mm.group(3) or "", 1)
            params["min_amount"] = v
        except ValueError:
            pass
    return {"term": term, "definition": definition, "params": params}
