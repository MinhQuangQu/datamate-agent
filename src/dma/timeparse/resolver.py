"""Deterministic resolver for Vietnamese time expressions -> inclusive date ranges.

Every interpretation is parameterised by TimeRules (compiled from the split's
rulebook), never hard-coded. The resolver produces CANDIDATES with provenance; the
planner (LLM) picks/combines them and the executor re-validates the chosen dates.

    tp = resolve("sp bán dc nhiều nhất tuần rồi là gì", date(2026, 3, 4), TimeRules(), cal)
    tp.periods[0] -> Period(2026-02-23, 2026-03-01, "last_week", "tuan roi")
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import date, timedelta

from dma.textnorm import nfc, strip_accents


@dataclass(frozen=True)
class TimeRules:
    week_start_isoday: int = 1                  # 1 = Monday ("một tuần = Thứ Hai -> Chủ nhật")
    current_period_to_date: bool = True         # "tháng này" = [đầu tháng, as_of]
    last_n_includes_today: bool = True          # "7 ngày qua" = [as_of-6, as_of]
    tet_start_lunar: tuple[int, int] = (12, 23)  # (month, day) of the PREVIOUS lunar year
    tet_end_lunar: tuple[int, int] = (1, 10)
    mid_autumn_lunar: tuple[int, int] = (8, 15)
    mid_autumn_window: tuple[int, int] = (0, 0)  # days before/after the 15th; rulebook decides
    weekend_isodays: tuple[int, ...] = (6, 7)
    bare_month_policy: str = "latest_not_after_as_of"   # or "current_year"
    mtd_compare_same_days: bool = True          # "tháng này so với tháng trước": same number of days
    shifts: tuple = ()                          # (("ca sang", 6, 13), ("ca chieu", 14, 21)) from the rulebook
    bare_day_policy: str = "latest_not_after_as_of"


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str
    text: str = ""
    note: str = ""

    def days(self) -> int:
        return (self.end - self.start).days + 1

    def as_json(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat(), "label": self.label,
                "text": self.text, "note": self.note}


@dataclass
class TimeParse:
    periods: list[Period] = field(default_factory=list)
    weekdays: tuple[int, ...] | None = None    # ISO weekdays filter ("cuối tuần", "thứ 7")
    flags: list[str] = field(default_factory=list)   # same_period_last_year, all_time, future, ...

    def as_json(self) -> dict:
        return {"periods": [p.as_json() for p in self.periods], "weekdays": self.weekdays, "flags": self.flags}


# ------------------------------------------------------------ period algebra
def _month_end(y: int, m: int) -> date:
    return (date(y + (m == 12), m % 12 + 1, 1) - timedelta(days=1))


def _shift_months(y: int, m: int, k: int) -> tuple[int, int]:
    idx = y * 12 + (m - 1) + k
    return idx // 12, idx % 12 + 1


def _safe_date(y: int, m: int, d: int) -> date:
    return date(y, m, min(d, _month_end(y, m).day))


def week_start(d: date, rules: TimeRules) -> date:
    return d - timedelta(days=(d.isoweekday() - rules.week_start_isoday) % 7)


def this_week(as_of: date, r: TimeRules) -> Period:
    s = week_start(as_of, r)
    return Period(s, as_of if r.current_period_to_date else s + timedelta(days=6), "this_week")


def last_week(as_of: date, r: TimeRules, n: int = 1) -> Period:
    s = week_start(as_of, r) - timedelta(days=7 * n)
    return Period(s, s + timedelta(days=6), "last_week" if n == 1 else f"week_minus_{n}")


def this_month(as_of: date, r: TimeRules) -> Period:
    s = as_of.replace(day=1)
    return Period(s, as_of if r.current_period_to_date else _month_end(as_of.year, as_of.month), "this_month")


def month(y: int, m: int, as_of: date | None = None, r: TimeRules | None = None) -> Period:
    end = _month_end(y, m)
    if as_of and r and r.current_period_to_date and (y, m) == (as_of.year, as_of.month):
        end = as_of
    return Period(date(y, m, 1), end, f"month_{y:04d}_{m:02d}")


def last_month(as_of: date, r: TimeRules, n: int = 1) -> Period:
    y, m = _shift_months(as_of.year, as_of.month, -n)
    p = month(y, m)
    return replace(p, label="last_month" if n == 1 else f"month_minus_{n}")


def quarter(y: int, q: int, as_of: date | None = None, r: TimeRules | None = None) -> Period:
    s = date(y, 3 * (q - 1) + 1, 1)
    e = _month_end(y, 3 * q)
    if as_of and r and r.current_period_to_date and s <= as_of <= e:
        e = as_of
    return Period(s, e, f"quarter_{y}_q{q}")


def this_quarter(as_of: date, r: TimeRules) -> Period:
    return replace(quarter(as_of.year, (as_of.month - 1) // 3 + 1, as_of, r), label="this_quarter")


def last_quarter(as_of: date, r: TimeRules) -> Period:
    q = (as_of.month - 1) // 3 + 1
    y, q = (as_of.year, q - 1) if q > 1 else (as_of.year - 1, 4)
    return replace(quarter(y, q), label="last_quarter")


def year(y: int, as_of: date | None = None, r: TimeRules | None = None) -> Period:
    e = date(y, 12, 31)
    if as_of and r and r.current_period_to_date and y == as_of.year:
        e = as_of
    return Period(date(y, 1, 1), e, f"year_{y}")


def last_n_days(as_of: date, n: int, r: TimeRules) -> Period:
    end = as_of if r.last_n_includes_today else as_of - timedelta(days=1)
    return Period(end - timedelta(days=n - 1), end, f"last_{n}_days")


def last_n_months_rolling(as_of: date, n: int, r: TimeRules) -> Period:
    y, m = _shift_months(as_of.year, as_of.month, -n)
    start = _safe_date(y, m, as_of.day) + timedelta(days=1)
    end = as_of if r.last_n_includes_today else as_of - timedelta(days=1)
    return Period(start, end, f"last_{n}_months_rolling", note="ambiguous: rolling vs full months")


def last_n_full_months(as_of: date, n: int) -> Period:
    y1, m1 = _shift_months(as_of.year, as_of.month, -n)
    y2, m2 = _shift_months(as_of.year, as_of.month, -1)
    return Period(date(y1, m1, 1), _month_end(y2, m2), f"last_{n}_full_months", note="ambiguous")


def same_period_last_year(p: Period) -> Period:
    def back(d: date) -> date:
        return _safe_date(d.year - 1, d.month, d.day)
    return Period(back(p.start), back(p.end), p.label + "_ly", p.text, "same period last year")


def tet_window(lunar_year: int, r: TimeRules, cal) -> Period:
    sm, sd = r.tet_start_lunar
    em, ed = r.tet_end_lunar
    start = cal.solar(sd, sm, lunar_year - 1 if sm > em else lunar_year)
    end = cal.solar(ed, em, lunar_year)
    return Period(start, end, f"tet_{lunar_year}")


def mid_autumn(lunar_year: int, r: TimeRules, cal) -> Period:
    m, d = r.mid_autumn_lunar
    day15 = cal.solar(d, m, lunar_year)
    b, a = r.mid_autumn_window
    return Period(day15 - timedelta(days=b), day15 + timedelta(days=a), f"mid_autumn_{lunar_year}")


# ------------------------------------------------------------- surface forms
_NUM_WORDS = {"mot": 1, "hai": 2, "ba": 3, "bon": 4, "tu": 4, "nam": 5, "sau": 6, "bay": 7,
              "tam": 8, "chin": 9, "muoi": 10, "mươi": 10}
_WEEKDAY = {"hai": 1, "2": 1, "ba": 2, "3": 2, "tu": 3, "4": 3, "nam": 4, "5": 4,
            "sau": 5, "6": 5, "bay": 6, "7": 6}
_PAST = r"(?:roi|truoc|trc|qua|vua roi|vua qua|vua xong)"
# "cùng kỳ (năm ngoái)", "cùng tháng năm trước", "cùng quý năm ngoái" — but NOT "cùng kỳ tháng trước" (month-to-date)
_SPLY = (r"\bcung (?:ky|thang|quy|dot)(?! (?:thang|th|tuan|quy)\b)(?: nay)?(?: (?:cua )?nam (?:ngoai|truoc|trc|roi))?\b")


def _plain(q: str) -> str:
    s = strip_accents(nfc(q).lower())
    return re.sub(r"\s+", " ", s)


def _n(tok: str) -> int | None:
    return int(tok) if tok.isdigit() else _NUM_WORDS.get(tok)


def resolve(question: str, as_of: date, rules: TimeRules, cal=None) -> TimeParse:
    """Return every time expression found in the question as a candidate period."""
    s = _plain(question)
    out = TimeParse()
    taken: list[tuple[int, int]] = []
    spans: list[tuple[int, int]] = []           # text span of out.periods[i] (scan phase only)

    def free(m: re.Match) -> bool:
        return all(m.end() <= a or m.start() >= b for a, b in taken)

    def add(m: re.Match, p: Period | None):
        taken.append((m.start(), m.end()))
        if p is not None:
            out.periods.append(replace(p, text=m.group(0).strip()))
            spans.append((m.start(), m.end()))

    def scan(pattern: str, fn):
        for m in re.finditer(pattern, s):
            if free(m):
                add(m, fn(m))

    # order matters: longest / most specific first
    scan(_SPLY, lambda m: out.flags.append("same_period_last_year"))
    scan(r"\b(tu truoc (toi|den) (nay|gio)|tu khi mo (cua|quan|tiem)|toan bo|tat ca cac (thang|nam))\b",
         lambda m: out.flags.append("all_time"))
    scan(r"\btuan truoc nua\b", lambda m: last_week(as_of, rules, 2))
    scan(r"\bthang truoc nua\b", lambda m: last_month(as_of, rules, 2))
    scan(r"\bnam kia\b", lambda m: year(as_of.year - 2))
    scan(r"\b(\d+|mot|hai|ba|bon|nam|sau|bay|tam|chin|muoi) (ngay|tuan|thang) (qua|gan day|gan nhat|vua qua|vua roi|tro lai day|truoc|cuoi)\b",
         lambda m: _last_n(m, as_of, rules))
    scan(r"\btu dau (tuan|thang|quy|nam)( (nay|hien tai))?( (den|toi) (nay|gio|hom nay))?\b",
         lambda m: {"tuan": this_week, "thang": this_month, "quy": this_quarter,
                    "nam": lambda a, r: replace(year(a.year, a, r), label="this_year")}[m.group(1)](as_of, rules))
    scan(r"\b(hom nay|hnay|bua nay|ngay nay)\b", lambda m: Period(as_of, as_of, "today"))
    scan(r"\b(hom qua|hqua|bua qua)\b", lambda m: Period(as_of - timedelta(1), as_of - timedelta(1), "yesterday"))
    scan(r"\bhom kia\b", lambda m: Period(as_of - timedelta(2), as_of - timedelta(2), "day_minus_2"))
    scan(rf"\btuan (nay|hien tai)\b", lambda m: this_week(as_of, rules))
    scan(rf"\btuan {_PAST}\b", lambda m: last_week(as_of, rules))
    # "th8, th9, th10/2025", "tháng 3, tháng 4, tháng 5 năm 2025": every month takes the trailing year
    for m in re.finditer(r"\b(?:thang|thg|th|t) ?(\d{1,2})((?:\s*(?:,|va|-)\s*(?:(?:thang|thg|th|t) ?)?\d{1,2})+)"
                         r"\s*(?:/|-|nam)\s*(\d{4})\b", s):
        if not free(m):
            continue
        y = int(m.group(3))
        mos = [int(m.group(1))] + [int(x) for x in re.findall(r"(\d{1,2})", m.group(2))]
        if all(1 <= x <= 12 for x in mos) and 2000 <= y <= 2100:
            taken.append((m.start(), m.end()))
            for x in mos:
                out.periods.append(replace(month(y, x, as_of, rules), text=m.group(0).strip()))
                spans.append((m.start(), m.end()))
    scan(r"\bthang (\d{1,2}) ?(?:/|-|\.|nam) ?(\d{4}|\d{2})\b(?! ?[/\-.] ?\d)",
         lambda m: _month_year(m, as_of, rules))
    scan(r"\b(?:t|th|thg) ?(\d{1,2}) ?[/\-] ?(\d{4}|\d{2})\b(?! ?[/\-.] ?\d)",
         lambda m: _month_year(m, as_of, rules))
    scan(r"\bthang (nay|hien tai)\b", lambda m: this_month(as_of, rules))
    scan(rf"\bthang {_PAST}\b", lambda m: last_month(as_of, rules))
    scan(r"\bquy (\d|i{1,3}|iv)(?:(?: ?[/\-.] ?| (?:nam )?)(\d{4}))?\b", lambda m: _quarter(m, as_of, rules))
    scan(r"\bquy (nay|hien tai)\b", lambda m: this_quarter(as_of, rules))
    scan(rf"\bquy {_PAST}\b", lambda m: last_quarter(as_of, rules))
    scan(r"\btet(?: nguyen dan)?(?: (nam nay|nam ngoai|nam truoc|nam roi|(\d{4})))?\b",
         lambda m: _tet(m, as_of, rules, cal))
    scan(r"\b(trung thu|ram thang tam)(?: (nam nay|nam ngoai|nam truoc|(\d{4})))?\b",
         lambda m: _mid_autumn(m, as_of, rules, cal))
    scan(r"\bnam (nay|hien tai)\b", lambda m: replace(year(as_of.year, as_of, rules), label="this_year"))
    scan(r"\bnam (ngoai|truoc|roi|qua|vua roi|vua qua)\b", lambda m: replace(year(as_of.year - 1), label="last_year"))
    scan(r"\bnam (\d{4})\b", lambda m: year(int(m.group(1)), as_of, rules))
    # bare "8/2025", "5-2026", "03/2026" (month/year) and "5/25", "3/26" (month/2-digit year; 25 is no month)
    scan(r"(?<![\d/.\-])(\d{1,2}) ?[/\-] ?(\d{4})\b(?! ?[/\-.] ?\d)", lambda m: _month_year(m, as_of, rules))
    scan(r"(?<![\d/.\-])(\d{1,2})/(\d{2})\b(?! ?[/\-.] ?\d)",
         lambda m: _month_year(m, as_of, rules) if 13 <= int(m.group(2)) <= 40 else None)
    scan(r"\b(?:ngay )?(\d{1,2}) ?[/\-.] ?(\d{1,2})(?: ?[/\-.] ?(\d{2,4}))?\b", lambda m: _day(m, as_of, rules))
    scan(r"\bthang (\d{1,2})\b", lambda m: _bare_month(m, as_of, rules))
    # weekday filters (not periods). "thứ hai" is also an ORDINAL ("bán chạy thứ hai"),
    # and "thu" appears in "doanh thu"/"trung thu": only explicit contexts count.
    wd: set[int] = set()
    for m in re.finditer(r"(?<!doanh )(?<!trung )\b(?:(vao|ngay|cac|moi|hom|nhung|toi|sang|chieu) )?"
                         r"thu (hai|ba|tu|nam|sau|bay|[2-7])\b( hang tuan| trong tuan)?", s):
        if not free(m):
            continue
        if m.group(1) or m.group(3):
            wd.add(_WEEKDAY[m.group(2)])
        else:
            out.flags.append(f"ambiguous_weekday_or_ordinal:{m.group(0).strip()}")
    if re.search(r"\bchu nhat\b", s):
        wd.add(7)
    if re.search(r"\bcuoi tuan\b", s):
        wd |= set(rules.weekend_isodays)
    if wd:
        out.weekdays = tuple(sorted(wd))
    _merge_ranges(s, out, spans)
    # "doanh thu tháng 5/2026 so với tháng trước (đó) / tháng liền trước / kỳ trước": the period BEFORE the named one
    named = [p for p in out.periods if re.match(r"(month|quarter|year)_\d{4}", p.label)]
    s_prev = re.sub(_SPLY, " ", s)            # "cùng kỳ năm trước" is not "kỳ trước"
    if len(named) == 1 and re.search(r"(thang|quy|nam|ky) (lien truoc|truoc do|truoc nua|truoc)\b|so voi (thang|quy|nam|ky) truoc", s_prev):
        x = named[0]
        if x.label.startswith("month_"):
            y, m = _shift_months(x.start.year, x.start.month, -1)
            prev = replace(month(y, m), label=f"prev_of_{x.label}")
        elif x.label.startswith("quarter_"):
            qn = (x.start.month - 1) // 3 + 1
            y, qn = (x.start.year, qn - 1) if qn > 1 else (x.start.year - 1, 4)
            prev = replace(quarter(y, qn), label=f"prev_of_{x.label}")
        else:
            prev = replace(year(x.start.year - 1), label=f"prev_of_{x.label}")
        out.periods = [p for p in out.periods if p.label not in ("last_month", "last_quarter", "last_year")] + \
            [replace(prev, text="ky truoc", note="period before the named one (rulebook: so với kỳ trước)")]
    # "tháng này ... (so với) tháng trước / cùng kỳ tháng trước": the previous month up to the same day number
    labels = [p.label for p in out.periods]
    if rules.mtd_compare_same_days and "this_month" in labels and (
            "last_month" in labels or re.search(r"cung ky (thang|th) (truoc|trc|roi)", s)):
        y, m = _shift_months(as_of.year, as_of.month, -1)
        lm = Period(date(y, m, 1), _safe_date(y, m, as_of.day), "last_month_same_days", "thang truoc",
                    "previous month up to the same day number (rulebook)")
        out.periods = [p for p in out.periods if p.label != "last_month"] + [lm]
    # "so với trước Tết": the same number of days right before the Tết window
    if re.search(r"truoc tet", s):
        tw = next((p for p in out.periods if p.label.startswith("tet_")), None)
        if tw is not None:
            n = tw.days()
            out.periods.append(Period(tw.start - timedelta(days=n), tw.start - timedelta(days=1), "before_" + tw.label,
                                      "truoc tet", f"{n} days before the Tết window"))
    if "same_period_last_year" in out.flags and out.periods:
        out.periods.append(same_period_last_year(out.periods[0]))
    for p in out.periods:
        if p.start > as_of:
            out.flags.append("future_period")
    return out


_RANGE_GAP = re.compile(r"\s*(den|toi|cho den|cho toi|-)( het| cuoi)?( ngay)?\s*")


def _merge_ranges(s: str, out: TimeParse, spans: list[tuple[int, int]]) -> None:
    """Compose "từ A (năm nay) đến (hết) B" into ONE inclusive range [A.start, B.end].

    Works on any pair of adjacent time expressions, so "từ 1/1 đến hết hôm nay", "từ đầu tháng này đến hết
    hôm nay" and "từ 3/5 tới 10/5" all go through the same rule instead of one regex each.
    """
    items = sorted(zip(spans, out.periods), key=lambda x: x[0][0])
    # a bare day directly followed by its year ("1/1 năm nay", "1/1 năm 2025") takes that year
    i = 0
    while i + 1 < len(items):
        (a, pa), (b, pb) = items[i], items[i + 1]
        if pa.label.startswith("day_") and re.fullmatch(r"(this_)?year(_\d{4})?", pb.label) \
                and not s[a[1]:b[0]].strip():
            try:
                d = pa.start.replace(year=pb.start.year)
            except ValueError:
                d = None
            if d is not None:
                items[i] = ((a[0], b[1]), replace(pa, start=d, end=d, label=f"day_{d.isoformat()}"))
                del items[i + 1]
                continue
        i += 1
    i = 0
    while i + 1 < len(items):
        (a, pa), (b, pb) = items[i], items[i + 1]
        opens = s[a[0]:a[1]].startswith("tu ") or re.search(r"\b(tu|bat dau tu) (ngay )?$", s[max(0, a[0] - 16):a[0]])
        if opens and _RANGE_GAP.fullmatch(s[a[1]:b[0]]) and pb.end >= pa.start:
            txt = s[a[0]:b[1]].strip()
            same = next((x for x in (pa, pb) if (x.start, x.end) == (pa.start, pb.end)), None)
            items[i] = ((a[0], b[1]), replace(same, text=txt) if same else
                        Period(pa.start, pb.end, "range", txt, f"from {pa.label} to {pb.label}"))
            del items[i + 1]
            continue
        i += 1
    seen: set = set()
    out.periods = []
    for _, p in items:                          # "từ 1/1 đến hết hôm nay, doanh thu năm nay": one period, not two
        if (p.start, p.end) not in seen:
            seen.add((p.start, p.end))
            out.periods.append(p)


def _last_n(m: re.Match, as_of: date, r: TimeRules) -> Period | None:
    n = _n(m.group(1))
    if not n:
        return None
    unit = m.group(2)
    if unit == "ngay":
        return last_n_days(as_of, n, r)
    if unit == "tuan":
        return replace(last_n_days(as_of, 7 * n, r), label=f"last_{n}_weeks_rolling", note="ambiguous: rolling vs full weeks")
    return last_n_months_rolling(as_of, n, r)


def _quarter(m: re.Match, as_of: date, r: TimeRules) -> Period | None:
    roman = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
    q = int(m.group(1)) if m.group(1).isdigit() else roman.get(m.group(1))
    if not q or not 1 <= q <= 4:
        return None
    y = int(m.group(2)) if m.group(2) else (as_of.year if 3 * (q - 1) + 1 <= as_of.month else as_of.year - 1)
    return quarter(y, q, as_of, r)


def _tet(m: re.Match, as_of: date, r: TimeRules, cal) -> Period | None:
    if cal is None:
        return None
    word = m.group(1) or ""
    if m.group(2):
        ly = int(m.group(2))
    elif word in ("nam ngoai", "nam truoc", "nam roi"):
        ly = as_of.year - 1
    elif word == "nam nay":
        ly = as_of.year
    else:  # bare "Tết": most recent window that has started by as_of
        ly = as_of.year if tet_window(as_of.year, r, cal).start <= as_of else as_of.year - 1
    return tet_window(ly, r, cal)


def _mid_autumn(m: re.Match, as_of: date, r: TimeRules, cal) -> Period | None:
    if cal is None:
        return None
    word = m.group(2) or ""
    if m.group(3):
        ly = int(m.group(3))
    elif word in ("nam ngoai", "nam truoc"):
        ly = as_of.year - 1
    elif word == "nam nay":
        ly = as_of.year
    else:
        ly = as_of.year if mid_autumn(as_of.year, r, cal).start <= as_of else as_of.year - 1
    return mid_autumn(ly, r, cal)


def _month_year(m: re.Match, as_of: date, r: TimeRules) -> Period | None:
    mo, y = int(m.group(1)), int(m.group(2))
    y = y + 2000 if y < 100 else y
    if not (1 <= mo <= 12 and 2000 <= y <= 2100):
        return None
    return month(y, mo, as_of, r)


def _day(m: re.Match, as_of: date, r: TimeRules) -> Period | None:
    d, mo = int(m.group(1)), int(m.group(2))
    if not (1 <= d <= 31 and 1 <= mo <= 12):
        return None
    if m.group(3):
        y = int(m.group(3))
        y = y + 2000 if y < 100 else y
    else:
        y = as_of.year
        if r.bare_day_policy == "latest_not_after_as_of" and date(y, mo, min(d, 28)) > as_of:
            y -= 1
    try:
        dd = date(y, mo, d)
    except ValueError:
        return None
    return Period(dd, dd, f"day_{dd.isoformat()}")


def _bare_month(m: re.Match, as_of: date, r: TimeRules) -> Period | None:
    mo = int(m.group(1))
    if not 1 <= mo <= 12:
        return None
    y = as_of.year
    if r.bare_month_policy == "latest_not_after_as_of" and mo > as_of.month:
        y -= 1
    return replace(month(y, mo, as_of, r), note=f"bare month -> {y} ({r.bare_month_policy})")
