from datetime import datetime

from dma.ingest.parsers import (US_FORMAT, VN_FORMAT, detect_date_order, detect_number_format, infer_column,
                                parse_bool, parse_date, parse_month, parse_number)


def v(s, fmt=None):
    p = parse_number(s, fmt)
    return None if p is None else p.value


def test_numbers_vietnamese_and_us():
    assert v("1.250.000 đ") == 1250000
    assert v("1,250,000") == 1250000
    assert v("1.250.000,5") == 1250000.5
    assert v("12,5") == 12.5
    assert v("(1.000.000)") == -1000000
    assert v("−3.200.000₫") == -3200000
    assert v("1 250 000") == 1250000
    assert v("abc") is None and v("") is None and v(None) is None


def test_colloquial_multipliers_units_percent():
    assert v("50k") == 50000
    assert v("1tr5") == 1500000
    assert v("2tr250") == 2250000
    assert v("1.5tr") == 1500000
    assert v("2 tỷ") == 2e9
    p = parse_number("12,45%")
    assert p.is_percent and p.value == 12.45
    q = parse_number("3 bao")
    assert q.value == 3 and q.unit == "bao"


def test_column_level_number_format():
    assert detect_number_format(["1.250.000", "35.000", "1.200"]) == VN_FORMAT
    assert detect_number_format(["1,250,000", "35,000.50"]) == US_FORMAT
    assert detect_number_format(["1.250", "2.500"], "Đơn giá") == VN_FORMAT
    assert v("1.250", VN_FORMAT) == 1250
    assert v("1.250", US_FORMAT) == 1.25
    assert detect_number_format(["2", "5", "10"]) is None


def test_dates():
    assert parse_date("05/03/2026") == datetime(2026, 3, 5)
    assert parse_date("05/03/2026", order="MDY") == datetime(2026, 5, 3)
    assert parse_date("2026-03-05 14:20") == datetime(2026, 3, 5, 14, 20)
    assert parse_date("ngày 5 tháng 3 năm 2026") == datetime(2026, 3, 5)
    assert parse_date(45720, allow_excel_serial=True) == datetime(2025, 3, 4)     # Excel serial
    assert parse_date(45720) is None
    assert detect_date_order(["13/02/2026", "01/02/2026"]) == "DMY"
    assert detect_date_order(["02/13/2026", "02/01/2026"]) == "MDY"
    assert parse_month("T3/2026") == (2026, 3) and parse_month("2025-09") == (2025, 9)


def test_bool():
    assert parse_bool("Có") is True and parse_bool("không") is False and parse_bool("x") is True
    assert parse_bool("1") is None and parse_bool("1", allow_01=True) is True


def test_infer_column():
    assert infer_column(["1.250.000", "35.000", None, ""], "Thành tiền").kind == "number"
    assert infer_column(["05/03/2026", "13/03/2026"], "Ngày bán").kind == "date"
    assert infer_column([45720, 45721.5], "Ngày").kind in ("date", "datetime")
    assert infer_column(["SP0001", "SP0042"], "Mã SP").kind == "id"
    assert infer_column(["Có", "Không", "Có"], "Đã trả hàng").kind == "bool"
    assert infer_column([1, 1, 1], "Số lượng").kind == "number"      # NOT bool
    assert infer_column(["2 bao", "5 bao"], "SL").units == ["bao"]
    assert infer_column(["Cà phê", "Trà"], "Tên món").kind == "string"
