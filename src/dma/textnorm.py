"""Text normalization shared by the scorer, NLU and entity linking.

Three distinct notions — never mix them up:
  * scorer_norm(): exact replica of the official string normalization (keeps diacritics).
  * fold():        accent-insensitive key, used ONLY for matching/lookup, never for output.
  * expand_abbrev(): conservative expansion of Vietnamese abbreviations / teencode.
"""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def scorer_norm(s: object) -> str:
    """Official order: 1) NFC  2) lowercase  3) every char that is not a letter,
    digit or whitespace -> one space  4) strip + collapse whitespace.
    Vietnamese diacritics are KEPT (so "dâu" != "dau")."""
    s = unicodedata.normalize("NFC", str(s)).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return _WS.sub(" ", s).strip()


def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", s.replace("đ", "d").replace("Đ", "D"))


def fold(s: object) -> str:
    """Accent-insensitive matching key: scorer_norm + strip accents."""
    return strip_accents(scorer_norm(s))


_VI_MARKS = set("ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ")


def has_vietnamese_diacritics(s: str) -> bool:
    return any(ch in _VI_MARKS for ch in nfc(s).lower())


# --- Abbreviations / teencode -------------------------------------------------
# Unambiguous tokens are expanded in the "expanded" text. Ambiguous tokens are NOT
# rewritten; they are reported as hints so the LLM decides from context.
UNAMBIGUOUS = {
    "sp": "sản phẩm", "mh": "mặt hàng", "sl": "số lượng", "ln": "lợi nhuận",
    "dh": "đơn hàng", "đh": "đơn hàng", "hd": "hóa đơn", "hđ": "hóa đơn",
    "ncc": "nhà cung cấp", "nv": "nhân viên", "ko": "không", "hok": "không",
    "hong": "không", "hông": "không", "khum": "không", "kô": "không",
    "dc": "được", "đc": "được", "dk": "được", "đk": "được", "j": "gì",
    "bn": "bao nhiêu", "bnhiu": "bao nhiêu", "bnh": "bao nhiêu", "ntn": "như thế nào",
    "thg": "tháng", "hnay": "hôm nay", "hqua": "hôm qua", "nhiu": "nhiêu",
    "tks": "cảm ơn", "vs": "với", "mn": "mọi người", "r": "rồi", "hh": "hàng hóa",
    "hsd": "hạn sử dụng", "nsx": "ngày sản xuất", "dvt": "đơn vị tính", "đvt": "đơn vị tính",
    "sdt": "số điện thoại", "sđt": "số điện thoại", "z": "vậy", "dzậy": "vậy", "cod": "thu hộ",
}
AMBIGUOUS = {
    "k": ["không", "nghìn (sau số, vd 50k)"],
    "kh": ["không", "khách hàng"],
    "kg": ["không", "kilogram"],
    "dt": ["doanh thu", "điện thoại"],
    "cn": ["chủ nhật", "chi nhánh"],
    "ck": ["chuyển khoản", "chiết khấu"],
    "tm": ["tiền mặt", "thương mại"],
    "tr": ["triệu", "trước"],
    "t2": ["thứ hai", "tháng 2"], "t3": ["thứ ba", "tháng 3"], "t4": ["thứ tư", "tháng 4"],
    "t5": ["thứ năm", "tháng 5"], "t6": ["thứ sáu", "tháng 6"], "t7": ["thứ bảy", "tháng 7"],
    "th": ["tháng", "thứ"],
    "tb": ["trung bình", "thông báo"],
    "tc": ["tổng cộng", "tiêu chuẩn"],
}
_TOKEN = re.compile(r"\w+", re.UNICODE)


def expand_abbrev(question: str) -> tuple[str, list[str]]:
    """Return (expanded_text, hints). Only whole tokens are touched; numbers like
    '50k' / '1tr5' are left for the numeric parser."""
    hints: list[str] = []
    out: list[str] = []
    pos = 0
    text = nfc(question)
    for m in _TOKEN.finditer(text):
        out.append(text[pos:m.start()])
        tok = m.group(0)
        low = tok.lower()
        if low in UNAMBIGUOUS:
            out.append(UNAMBIGUOUS[low])
        else:
            out.append(tok)
            if low in AMBIGUOUS:
                hints.append(f"'{tok}' có thể là: " + " | ".join(AMBIGUOUS[low]))
        pos = m.end()
    out.append(text[pos:])
    return _WS.sub(" ", "".join(out)).strip(), hints
