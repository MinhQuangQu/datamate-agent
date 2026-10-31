from datetime import date, timedelta

from dma.timeparse import lunar
from dma.timeparse.calendar import LunarCalendar
from dma.timeparse.resolver import TimeRules, resolve, tet_window

TET = {2021: date(2021, 2, 12), 2022: date(2022, 2, 1), 2023: date(2023, 1, 22),
       2024: date(2024, 2, 10), 2025: date(2025, 1, 29), 2026: date(2026, 2, 17)}


def test_tet_dates():
    for y, d in TET.items():
        assert lunar.tet_date(y) == d
        assert lunar.solar_to_lunar(d) == lunar.LunarDate(1, 1, y, False)


def test_mid_autumn():
    assert lunar.lunar_to_solar(15, 8, 2024) == date(2024, 9, 17)
    assert lunar.lunar_to_solar(15, 8, 2025) == date(2025, 10, 6)


def test_round_trip_2020_2028_incl_leap_months():
    d, leaps = date(2020, 1, 1), 0
    while d <= date(2028, 12, 31):
        ld = lunar.solar_to_lunar(d)
        leaps += ld.leap
        assert lunar.lunar_to_solar(ld.day, ld.month, ld.year, ld.leap) == d
        d += timedelta(days=1)
    assert leaps > 0   # 2020 (tháng 4 nhuận), 2023 (tháng 2 nhuận), 2025 (tháng 6 nhuận)


def test_calendar_csv_detection(tmp_path):
    p = tmp_path / "calendar.csv"
    rows = ["ngay_duong,ngay_am,thang_am,nam_am,nhuan,ten_le"]
    d = date(2025, 12, 1)
    while d <= date(2026, 3, 31):
        ld = lunar.solar_to_lunar(d)
        ev = "Tết Nguyên Đán" if (ld.month, ld.day) == (1, 1) else ""
        rows.append(f"{d.isoformat()},{ld.day},{ld.month},{ld.year},{int(ld.leap)},{ev}")
        d += timedelta(days=1)
    p.write_text("\n".join(rows), encoding="utf-8")
    cal = LunarCalendar.from_csv(str(p))
    assert cal.report.solar_col == "ngay_duong"
    assert cal.report.agreement_with_algorithm == 1.0
    assert cal.tet(2026) == date(2026, 2, 17)
    assert cal.find_event_dates("tet") == [date(2026, 2, 17)]


AS_OF = date(2026, 3, 4)            # Wednesday
R = TimeRules()
CAL = LunarCalendar.from_algorithm(date(2023, 1, 1), date(2027, 12, 31))


def first(q, as_of=AS_OF):
    tp = resolve(q, as_of, R, CAL)
    return (tp.periods[0].start, tp.periods[0].end) if tp.periods else None


def test_relative_periods():
    assert first("sp bán dc nhiều nhất tuần rồi là gì") == (date(2026, 2, 23), date(2026, 3, 1))
    assert first("thang nay hang nao ban cham nhat") == (date(2026, 3, 1), date(2026, 3, 4))
    assert first("doanh thu tháng trước") == (date(2026, 2, 1), date(2026, 2, 28))
    assert first("7 ngày qua bán bao nhiêu ly") == (date(2026, 2, 26), date(2026, 3, 4))
    assert first("hôm qua") == (date(2026, 3, 3), date(2026, 3, 3))
    assert first("tuần này") == (date(2026, 3, 2), date(2026, 3, 4))
    assert first("tuần trước nữa") == (date(2026, 2, 16), date(2026, 2, 22))
    assert first("năm ngoái") == (date(2025, 1, 1), date(2025, 12, 31))
    assert first("quý 4 năm 2025") == (date(2025, 10, 1), date(2025, 12, 31))
    assert first("quý trước") == (date(2025, 10, 1), date(2025, 12, 31))
    assert first("tháng 3/2025") == (date(2025, 3, 1), date(2025, 3, 31))
    assert first("tháng 12") == (date(2025, 12, 1), date(2025, 12, 31))     # bare month, latest ≤ as_of
    assert first("từ đầu năm tới nay") == (date(2026, 1, 1), date(2026, 3, 4))


def test_same_period_last_year_and_holidays():
    tp = resolve("doanh thu tháng này so với cùng kỳ năm ngoái", AS_OF, R, CAL)
    assert "same_period_last_year" in tp.flags
    assert (tp.periods[-1].start, tp.periods[-1].end) == (date(2025, 3, 1), date(2025, 3, 4))
    assert first("doanh thu dịp Tết năm nay") == (date(2026, 2, 10), date(2026, 2, 26))
    w = tet_window(2025, R, CAL)
    assert (w.start, w.end) == (date(2025, 1, 22), date(2025, 2, 7))
    assert first("trung thu năm ngoái bán bao nhiêu hộp bánh") == (date(2025, 10, 6), date(2025, 10, 6))


def test_weekday_filters_and_ambiguity():
    assert resolve("tháng này bán vào thứ 7 nhiều không", AS_OF, R, CAL).weekdays == (6,)
    assert resolve("cuối tuần rồi", AS_OF, R, CAL).weekdays == (6, 7)
    tp = resolve("doanh thu ba tháng gần đây", AS_OF, R, CAL)
    assert tp.weekdays is None and tp.periods[0].label == "last_3_months_rolling"
    tp = resolve("sản phẩm bán chạy thứ hai tháng này", AS_OF, R, CAL)
    assert tp.weekdays is None and any(f.startswith("ambiguous_weekday_or_ordinal") for f in tp.flags)
    tp = resolve("trung thu năm nay", date(2026, 10, 1), R, CAL)
    assert tp.weekdays is None


def test_from_to_ranges_compose_into_one_period():
    assert first("Tính từ ngày 1/1 đến hết hôm nay, doanh thu năm nay là bao nhiêu?") == (date(2026, 1, 1), AS_OF)
    assert first("Từ 1/1 năm nay đến hết hôm nay đã bán được bao nhiêu đơn?") == (date(2026, 1, 1), AS_OF)
    assert first("Từ đầu năm đến hết hôm nay bán được bao nhiêu tiền?") == (date(2026, 1, 1), AS_OF)
    assert first("doanh thu từ 3/2 tới 10/2") == (date(2026, 2, 3), date(2026, 2, 10))
    tp = resolve("dthu tu dau thang nay den het hnay tang hay giam bn % so voi cung ky thang trc", AS_OF, R, CAL)
    assert [(p.start, p.end) for p in tp.periods] == [(date(2026, 3, 1), AS_OF), (date(2026, 2, 1), date(2026, 2, 4))]


def test_same_period_last_year_is_not_previous_period():
    tp = resolve("so cùng kỳ năm trước, doanh thu tháng 2/2026 tăng giảm mấy %", AS_OF, R, CAL)
    assert [p.start for p in tp.periods] == [date(2026, 2, 1), date(2025, 2, 1)]
    tp = resolve("so vs cung thang nam truoc, dthu thang 1 nam 2026 thay doi may %", AS_OF, R, CAL)
    assert [p.start for p in tp.periods] == [date(2026, 1, 1), date(2025, 1, 1)]
    tp = resolve("doanh thu tháng 2/2026 so với tháng trước", AS_OF, R, CAL)
    assert [p.start for p in tp.periods] == [date(2026, 2, 1), date(2026, 1, 1)]


def test_quarter_with_slash_year():
    assert first("Tổng doanh thu quý 1/2025 là bao nhiêu?") == (date(2025, 1, 1), date(2025, 3, 31))
    assert first("so don quy 3/2025 la bn") == (date(2025, 7, 1), date(2025, 9, 30))
