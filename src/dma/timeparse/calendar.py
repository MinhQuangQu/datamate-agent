"""Split-level lunar/solar calendar loaded from calendar.csv (source of truth).

The exact layout of calendar.csv is not documented by the organisers, so columns are
detected heuristically and the result is cross-checked against the astronomical
algorithm in lunar.py. Detection results are logged in `self.report` — inspect them
on day 1 of EDA and pin the mapping in configs if needed.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from dma.ingest.parsers import parse_bool, parse_date
from dma.textnorm import fold
from dma.timeparse import lunar as algo
from dma.timeparse.lunar import LunarDate


@dataclass
class CalendarReport:
    source: str = "algorithm"
    solar_col: str | None = None
    lunar_cols: dict[str, str] = field(default_factory=dict)
    event_cols: list[str] = field(default_factory=list)
    rows: int = 0
    agreement_with_algorithm: float | None = None
    warnings: list[str] = field(default_factory=list)


class LunarCalendar:
    def __init__(self, s2l: dict[date, LunarDate], events: dict[date, list[str]], report: CalendarReport):
        self._s2l = s2l
        self._l2s = {(l.year, l.month, l.day, l.leap): d for d, l in s2l.items()}
        self._events = events
        self.report = report

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_algorithm(cls, start: date = date(2018, 1, 1), end: date = date(2030, 12, 31)) -> "LunarCalendar":
        return cls(algo.lunar_range(start, end), {}, CalendarReport(source="algorithm"))

    @classmethod
    def from_csv(cls, path: str) -> "LunarCalendar":
        with open(path, "rb") as f:
            raw = f.read()
        for enc in ("utf-8-sig", "utf-8", "cp1258", "utf-16"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("cannot decode calendar.csv")
        sample = text[:4096]
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        rows = list(csv.reader(io.StringIO(text), dialect))
        header, body = rows[0], [r for r in rows[1:] if any(c.strip() for c in r)]
        return cls._from_table(header, body)

    @classmethod
    def _from_table(cls, header: list[str], body: list[list[str]]) -> "LunarCalendar":
        rep = CalendarReport(source="calendar.csv", rows=len(body))
        names = [fold(h) for h in header]
        col = {h: [r[i] if i < len(r) else "" for r in body] for i, h in enumerate(header)}

        def is_lunar(n: str) -> bool:
            return bool(re.search(r"\b(am|lunar|al)\b", n)) or n.startswith("am ") or " am" in n

        def parses_as_dates(vals: list[str]) -> float:
            vals = [v for v in vals[:200] if v.strip()]
            return sum(parse_date(v) is not None for v in vals) / max(1, len(vals))

        solar = None
        for h, n in zip(header, names):
            if not is_lunar(n) and parses_as_dates(col[h]) > 0.95 and \
                    any(k in n for k in ("duong", "solar", "date", "ngay", "dl")):
                solar = h
                break
        if solar is None:  # any non-lunar full-date column
            solar = next((h for h, n in zip(header, names) if not is_lunar(n) and parses_as_dates(col[h]) > 0.95), None)
        if solar is None:
            raise ValueError(f"no solar-date column found in calendar header {header}")
        rep.solar_col = solar

        lunar_day = lunar_month = lunar_year = lunar_leap = lunar_full = None
        for h, n in zip(header, names):
            if h == solar:
                continue
            lun = is_lunar(n)
            if lun and re.search(r"\b(ngay|day|d)\b", n) and lunar_day is None:
                lunar_day = h
            elif lun and re.search(r"\b(thang|month|m)\b", n) and lunar_month is None:
                lunar_month = h
            elif lun and re.search(r"\b(nam|year|y)\b", n) and lunar_year is None:
                lunar_year = h
            elif re.search(r"(nhuan|leap)", n) and lunar_leap is None:
                lunar_leap = h
            elif lun and lunar_full is None and any(re.match(r"^\d{1,4}[/-]\d{1,2}([/-]\d{1,4})?$", v.strip())
                                                    for v in col[h][:20]):
                lunar_full = h
        for k, v in (("day", lunar_day), ("month", lunar_month), ("year", lunar_year),
                     ("leap", lunar_leap), ("full", lunar_full)):
            if v:
                rep.lunar_cols[k] = v
        rep.event_cols = [h for h, n in zip(header, names)
                          if h not in (solar, lunar_day, lunar_month, lunar_year, lunar_leap, lunar_full)
                          and any(k in n for k in ("le", "su kien", "holiday", "event", "ghi chu", "dip", "ten", "mua"))]

        s2l: dict[date, LunarDate] = {}
        events: dict[date, list[str]] = {}
        for i in range(len(body)):
            sd = parse_date(col[solar][i])
            if sd is None:
                continue
            d = sd.date()
            ld = None
            leap = bool(parse_bool(col[lunar_leap][i], allow_01=True)) if lunar_leap else False
            try:
                if lunar_day and lunar_month:
                    ly = int(float(col[lunar_year][i])) if lunar_year else None
                    lm, ldd = int(float(col[lunar_month][i])), int(float(col[lunar_day][i]))
                    if ly is None:
                        ly = d.year - 1 if (lm >= 11 and d.month <= 3) else d.year
                    ld = LunarDate(ldd, lm, ly, leap)
                elif lunar_full:
                    parts = [int(p) for p in re.split(r"[/-]", col[lunar_full][i].strip())]
                    if len(parts) == 3 and parts[0] > 31:          # yyyy-mm-dd
                        ld = LunarDate(parts[2], parts[1], parts[0], leap)
                    elif len(parts) == 3:                           # dd/mm/yyyy
                        ld = LunarDate(parts[0], parts[1], parts[2], leap)
                    elif len(parts) == 2:                           # dd/mm
                        ly = d.year - 1 if (parts[1] >= 11 and d.month <= 3) else d.year
                        ld = LunarDate(parts[0], parts[1], ly, leap)
            except (ValueError, TypeError):
                ld = None
            if ld is not None:
                s2l[d] = ld
            ev = [col[h][i].strip() for h in rep.event_cols if col[h][i].strip() and fold(col[h][i]) not in ("0", "false", "khong")]
            if ev:
                events[d] = ev
        if not s2l:
            rep.warnings.append("no lunar columns parsed; falling back to algorithm for lunar dates")
            dates = [parse_date(v) for v in col[solar]]
            ds = [x.date() for x in dates if x]
            s2l = algo.lunar_range(min(ds), max(ds))
            rep.source = "calendar.csv(events)+algorithm(lunar)"
        cal = cls(s2l, events, rep)
        rep.agreement_with_algorithm = cal.cross_check()
        if rep.agreement_with_algorithm is not None and rep.agreement_with_algorithm < 0.99:
            rep.warnings.append(f"calendar.csv disagrees with algorithm on {1 - rep.agreement_with_algorithm:.1%} "
                                "of days — calendar.csv wins (it is what the reference used)")
        return cal

    # --------------------------------------------------------------- queries
    def lunar(self, d: date) -> LunarDate:
        return self._s2l.get(d) or algo.solar_to_lunar(d)

    def solar(self, day: int, month: int, year: int, leap: bool = False) -> date | None:
        hit = self._l2s.get((year, month, day, leap))
        # outside the csv range (or day 30 of a 29-day month): fall back to the algorithm
        return hit or algo.lunar_to_solar(day, month, year, leap)

    def tet(self, lunar_year: int) -> date:
        d = self.solar(1, 1, lunar_year)
        assert d is not None
        return d

    def events_on(self, d: date) -> list[str]:
        return self._events.get(d, [])

    def find_event_dates(self, keyword: str, year: int | None = None) -> list[date]:
        k = fold(keyword)
        return sorted(d for d, evs in self._events.items()
                      if (year is None or d.year == year) and any(k in fold(e) for e in evs))

    def cross_check(self, sample_every: int = 7) -> float | None:
        items = sorted(self._s2l.items())[::sample_every]
        if not items:
            return None
        ok = sum(algo.solar_to_lunar(d) == l for d, l in items)
        return ok / len(items)

    def date_range(self) -> tuple[date, date] | None:
        if not self._s2l:
            return None
        ks = sorted(self._s2l)
        return ks[0], ks[-1]


def daterange(a: date, b: date):
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)
