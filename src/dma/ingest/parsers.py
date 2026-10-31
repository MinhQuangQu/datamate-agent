"""Vietnamese-aware value parsers with column-level format inference.

Real SME spreadsheets mix: "1.250.000 đ", "1,250,000", "1tr5", "50k", "(1.000)",
"2 bao", "05/03/2026", "2026-03-05 14:20", Excel serials (45720), "có/không", "x"...
A single cell is ambiguous ("1.250" = 1250 or 1.25?), so we decide the format per
COLUMN first (detect_number_format / detect_date_order), then parse every cell.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Iterable, Sequence

from dma.textnorm import fold, nfc, strip_accents


def _plain(s: object) -> str:
    """lowercase + no accents, punctuation KEPT (unlike fold())."""
    return strip_accents(nfc(str(s)).lower()).strip()


# ----------------------------------------------------------------- numbers
_CURRENCY = re.compile(r"\s*(vnđ|vnd|đồng|dong|₫|đ|usd|\$)\s*$|^\s*(vnđ|vnd|₫|\$)\s*", re.I)
_MULT = {"k": 1e3, "nghìn": 1e3, "nghin": 1e3, "ngàn": 1e3, "ngan": 1e3,
         "tr": 1e6, "triệu": 1e6, "trieu": 1e6, "củ": 1e6, "cu": 1e6,
         "tỷ": 1e9, "tỉ": 1e9, "ty": 1e9, "ti": 1e9}
_MULT_ALT = "|".join(sorted(map(re.escape, _MULT), key=len, reverse=True))
_COLLOQ = re.compile(rf"^(\d+)\s*({_MULT_ALT})\s*(\d+)$", re.I)        # 1tr5, 2tr250, 1k2
_SUFFIX = re.compile(rf"^([\d.,\s]+?)\s*({_MULT_ALT})$", re.I)         # 50k, 1.5tr, 2 tỷ
UNITS = {"bao", "thung", "hop", "chai", "lon", "ly", "coc", "kg", "g", "goi", "cai", "chiec",
         "vien", "vi", "tuyp", "lo", "ml", "l", "lit", "m", "m2", "m3", "tan", "ta", "cay", "cuon",
         "bo", "doi", "phan", "suat", "dia", "to", "bat", "con", "qua", "trai", "bo hoa", "kien",
         "thanh", "tui", "can", "binh", "ong", "goi", "set", "combo", "lan", "buoi", "gio", "dem",
         "loc", "cuc", "khay", "hu", "cap", "bich", "xau", "chuc", "tep", "mieng", "ve", "bich", "kep", "thanh", "que",
         "hu", "ram", "tam", "soi", "cuon", "bom", "chau", "cut", "khoi", "manh", "pho", "nai", "buong", "bo"}
_WITH_UNIT = re.compile(r"^([\d.,]+)\s*([a-z][a-z0-9 ]{0,8})$")
_VN_THOUS = re.compile(r"^\d{1,3}(\.\d{3})+(,\d+)?$")
_US_THOUS = re.compile(r"^\d{1,3}(,\d{3})+(\.\d+)?$")
_SPACE_THOUS = re.compile(r"^\d{1,3}( \d{3})+([.,]\d+)?$")


@dataclass(frozen=True)
class NumberFormat:
    decimal: str = "."            # "." or ","
    thousands: str | None = None  # ".", ",", " " or None


VN_FORMAT = NumberFormat(decimal=",", thousands=".")
US_FORMAT = NumberFormat(decimal=".", thousands=",")


@dataclass
class ParsedNumber:
    value: float
    is_percent: bool = False
    had_currency: bool = False
    multiplier: float = 1.0
    unit: str | None = None


def _core_to_float(core: str, fmt: NumberFormat | None) -> float | None:
    core = core.replace(" ", " ").strip()
    if _SPACE_THOUS.match(core):
        core = core.replace(" ", "")
    if not core or not re.fullmatch(r"[\d.,]+", core) or not re.search(r"\d", core):
        return None
    if fmt is not None:
        th, dec = fmt.thousands, fmt.decimal
        if th and th in core:
            int_part = core.split(dec)[0] if dec in core else core
            groups = int_part.split(th)
            if 1 <= len(groups[0]) <= 3 and all(len(g) == 3 for g in groups[1:]):
                core = core.replace(th, "")
            elif core.count(th) == 1 and dec not in core:
                core = core.replace(th, dec)          # "12.5" inside a VN column -> 12,5
            else:
                return None
        if dec != "." and "." in core:
            return None
        core = core.replace(dec, ".")
        return float(core) if core.count(".") <= 1 else None
    # no column-level format: infer from the string alone
    if "." in core and "," in core:
        dec = "." if core.rfind(".") > core.rfind(",") else ","
        th = "," if dec == "." else "."
        core = core.replace(th, "").replace(dec, ".")
        return float(core) if core.count(".") <= 1 else None
    if _VN_THOUS.match(core) and core.count(".") >= 2:
        return float(core.replace(".", ""))
    if _US_THOUS.match(core) and core.count(",") >= 2:
        return float(core.replace(",", ""))
    if core.count(",") == 1:
        return float(core.replace(",", "."))          # VN decimal comma; "1,250" stays ambiguous
    if core.count(".") <= 1:
        return float(core)
    return None


def parse_number(raw: object, fmt: NumberFormat | None = None, allow_unit: bool = True) -> ParsedNumber | None:
    """Parse one cell or phrase into a number. Returns None if it is not numeric."""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return None if (isinstance(raw, float) and math.isnan(raw)) else ParsedNumber(float(raw))
    s = nfc(str(raw)).strip().lower().replace("−", "-").replace("–", "-")
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg, s = True, s[1:-1].strip()
    if s.startswith("-"):
        neg, s = True, s[1:].strip()
    elif s.startswith("+"):
        s = s[1:].strip()
    is_pct = s.endswith("%")
    if is_pct:
        s = s[:-1].strip()
    had_cur = bool(_CURRENCY.search(s))
    s = _CURRENCY.sub("", s).strip()
    m = _COLLOQ.match(s)
    if m:
        base, unit, frac = m.groups()
        v = (float(base) + float("0." + frac)) * _MULT[unit.lower()]
        return ParsedNumber(-v if neg else v, is_pct, had_cur, _MULT[unit.lower()])
    mult, unit_word = 1.0, None
    m = _SUFFIX.match(s)
    if m:
        s, mult = m.group(1).strip(), _MULT[m.group(2).lower()]
    elif allow_unit:
        m = _WITH_UNIT.match(_plain(s))
        if m and m.group(2).strip().split()[0] in UNITS:      # "5 lốc", "1 bao 25kg", "2 hộp 30"
            s, unit_word = m.group(1), m.group(2).strip()
    try:
        v = _core_to_float(s, fmt)
    except ValueError:
        v = None
    if v is None:
        return None
    v *= mult
    return ParsedNumber(-v if neg else v, is_pct, had_cur, mult, unit_word)


_MONEY_HINT = ("tien", "gia", "don gia", "amount", "price", "total", "doanh thu", "chi phi",
               "cong no", "luong", "phi", "von", "thanh toan", "giam gia", "chiet khau")


def detect_number_format(values: Iterable[object], column_name: str = "") -> NumberFormat | None:
    """Vote over a column. None = no separator evidence (plain integers / floats)."""
    votes: Counter = Counter()
    single_dot_groups = 0
    for raw in values:
        if raw is None or isinstance(raw, (int, float)):
            continue
        s = _CURRENCY.sub("", nfc(str(raw)).strip().lower()).strip()
        s = re.sub(r"^[(\-+]+|[)%]+$", "", s).strip()
        s = re.sub(r"\s*[a-zà-ỹđ]+$", "", s)          # drop unit words: "2 bao" -> "2"
        if not s:
            continue
        if re.fullmatch(r"\d{1,3}(\.\d{3}){2,}(,\d+)?", s) or re.fullmatch(r"\d{1,3}(\.\d{3})+,\d+", s):
            votes["vn"] += 1
        elif re.fullmatch(r"\d{1,3}(,\d{3}){2,}(\.\d+)?", s) or re.fullmatch(r"\d{1,3}(,\d{3})+\.\d+", s):
            votes["us"] += 1
        elif re.fullmatch(r"\d+,(\d{1,2}|\d{4,})", s):
            votes["vn"] += 1                           # decimal comma: "12,5"
        elif re.fullmatch(r"\d+\.(\d{1,2}|\d{4,})", s):
            votes["us"] += 1                           # decimal dot: "12.75"
        elif re.fullmatch(r"\d{1,3}\.\d{3}", s):
            single_dot_groups += 1                     # "1.250": ambiguous
        elif re.fullmatch(r"\d{1,3},\d{3}", s):
            votes["us_weak"] += 1
    if votes["vn"] > votes["us"]:
        return VN_FORMAT
    if votes["us"] > votes["vn"]:
        return US_FORMAT
    if single_dot_groups and (any(h in fold(column_name) for h in _MONEY_HINT) or not votes["us_weak"]):
        return VN_FORMAT
    if votes["us_weak"]:
        return US_FORMAT
    return None


# ------------------------------------------------------------------- dates
_EXCEL_EPOCH = datetime(1899, 12, 30)
_ISO = re.compile(r"^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ t](\d{1,2}):(\d{2})(?::(\d{2}))?)?")
_DMY = re.compile(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})(?:[ ,]+(\d{1,2})[:h](\d{2})(?::(\d{2}))?)?")
_VI_TEXT = re.compile(r"(?:ngay\s*)?(\d{1,2})\s*(?:thang|thg)\s*(\d{1,2})\s*(?:nam|,)?\s*(\d{4})")
_MONTH_ONLY = re.compile(r"^(?:(?:thang|thg|t)\s*)?(\d{1,2})\s*[-/.]\s*(\d{4})$|^(\d{4})[-/.](\d{1,2})$")


def _mk(y: int, m: int, d: int, hh=None, mi=None, ss=None) -> datetime | None:
    try:
        if y < 100:
            y += 2000
        return datetime(y, m, d, int(hh or 0), int(mi or 0), int(ss or 0))
    except ValueError:
        return None


def detect_date_order(values: Iterable[object]) -> str:
    """'DMY' (Vietnamese default) or 'MDY', decided by unambiguous cells."""
    dmy = mdy = 0
    for raw in values:
        if raw is None:
            continue
        m = _DMY.match(str(raw).strip())
        if not m:
            continue
        a, b = int(m.group(1)), int(m.group(2))
        if a > 12 >= b:
            dmy += 1
        elif b > 12 >= a:
            mdy += 1
    return "MDY" if mdy > dmy else "DMY"


def parse_date(raw: object, order: str = "DMY", allow_excel_serial: bool = False) -> datetime | None:
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, date):
        return datetime(raw.year, raw.month, raw.day)
    if isinstance(raw, (int, float)):
        if allow_excel_serial and not (isinstance(raw, float) and math.isnan(raw)) and 20000 <= raw <= 80000:
            return _EXCEL_EPOCH + timedelta(days=float(raw))
        return None
    s = _plain(raw)
    if not s:
        return None
    m = _ISO.match(s)
    if m:
        return _mk(int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5), m.group(6))
    m = _DMY.match(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        d, mo = (a, b) if order == "DMY" else (b, a)
        return _mk(y, mo, d, m.group(4), m.group(5), m.group(6))
    m = _VI_TEXT.search(s)
    if m:
        return _mk(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    if allow_excel_serial and re.fullmatch(r"\d{5}(\.\d+)?", s):
        return parse_date(float(s), order, True)
    return None


def parse_month(raw: object) -> tuple[int, int] | None:
    """'03/2026', 'T3/2026', 'tháng 3-2026', '2026-03' -> (2026, 3)."""
    m = _MONTH_ONLY.match(_plain(raw))
    if not m:
        return None
    if m.group(1):
        mo, y = int(m.group(1)), int(m.group(2))
    else:
        y, mo = int(m.group(3)), int(m.group(4))
    return (y, mo) if 1 <= mo <= 12 else None


# ---------------------------------------------------------------- booleans
_TRUE = {"co", "x", "✓", "✔", "yes", "y", "true", "da", "roi", "dung", "ok"}
_FALSE = {"khong", "ko", "no", "n", "false", "chua", "sai"}
_FLAG_NAME_HINT = ("is ", "co ", "da ", "huy", "tra", "flag", "returned", "cancel", "vip")


def parse_bool(raw: object, allow_01: bool = False) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = fold(raw)
    if s in _TRUE or (allow_01 and s == "1"):
        return True
    if s in _FALSE or (allow_01 and s == "0"):
        return False
    return None


# ------------------------------------------------------- column inference
_ID_PATTERN = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{1,8}$")
_DATE_NAME_HINT = ("ngay", "date", "thoi gian", "time", "han", "hsd", "exp", "nsx", "mfg", "thang")


@dataclass
class ColumnInference:
    kind: str                         # empty|bool|date|datetime|number|percent|id|string
    parsed: list = field(default_factory=list)
    parse_rate: float = 0.0
    number_format: NumberFormat | None = None
    date_order: str | None = None
    units: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _nonempty(values: Sequence[object]) -> list[object]:
    out = []
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()):
            continue
        out.append(v)
    return out


def infer_column(values: Sequence[object], name: str = "", threshold: float = 0.95) -> ColumnInference:
    vals = _nonempty(values)
    if not vals:
        return ColumnInference("empty", [None] * len(values), 1.0)
    n = len(vals)
    flag_name = any(h in fold(name) + " " for h in _FLAG_NAME_HINT)
    distinct = {fold(v) for v in vals}
    # bool: tiny vocabulary; numeric 0/1 only when the header looks like a flag
    if len(distinct) <= 3 and all(parse_bool(v, allow_01=flag_name) is not None for v in vals) \
            and (flag_name or not distinct <= {"0", "1"}):
        return ColumnInference("bool", [parse_bool(v, allow_01=flag_name) for v in values], 1.0)
    if sum(bool(_ID_PATTERN.match(str(v).strip())) for v in vals) >= threshold * n:
        return ColumnInference("id", [None if v is None else str(v).strip() for v in values], 1.0)
    date_name = any(h in fold(name) for h in _DATE_NAME_HINT)
    order = detect_date_order(vals)
    # numeric cells only count as dates (Excel serials) when the header looks like a date
    ok_dates = sum(parse_date(v, order, allow_excel_serial=date_name) is not None for v in vals)
    if ok_dates >= threshold * n:
        parsed = [parse_date(v, order, allow_excel_serial=date_name) for v in values]
        has_time = any(d is not None and (d.hour or d.minute or d.second) for d in parsed)
        return ColumnInference("datetime" if has_time else "date", parsed, ok_dates / n, date_order=order)
    fmt = detect_number_format(vals, name)
    nums = [parse_number(v, fmt) for v in vals]
    ok = [x for x in nums if x is not None]
    if len(ok) >= threshold * n:
        pct = sum(x.is_percent for x in ok) >= 0.5 * n
        parsed = [(lambda p: None if p is None else p.value)(parse_number(v, fmt)) for v in values]
        ci = ColumnInference("percent" if pct else "number", parsed, len(ok) / n, number_format=fmt,
                             units=sorted({x.unit for x in ok if x.unit}))
        if fmt is None and any(re.fullmatch(r"\d{1,3},\d{3}", str(v)) for v in vals):
            ci.notes.append("ambiguous '1,250' cells parsed as decimal comma")
        return ci
    return ColumnInference("string", [None if v is None else nfc(str(v)).strip() for v in values], 1.0)
