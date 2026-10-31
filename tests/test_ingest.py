"""Ingestion edge cases E1–E16 (05.1 §12)."""
import sqlite3

import duckdb
import openpyxl

from dma.ingest.loader import ingest_business
from dma.ingest.tables import detect_header, grid_to_tables, split_blocks


def _typed(res, name):
    con = duckdb.connect(res.db_path, read_only=True)
    try:
        return con.execute(f'SELECT * FROM typed."{name}"').fetchall()
    finally:
        con.close()


def test_csv_bom_semicolon_preamble_total_units(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "ban.csv").write_text("BÁO CÁO BÁN HÀNG;;;\nĐVT: đồng;;;\n\nMã HĐ;Ngày;SL;Thành tiền\n"
                               "HD1;05/03/2026;2 bao;1.250.000\nHD2;13/03/2026;1;(450.000)\n"
                               "Tổng cộng;;3;800.000\n", encoding="utf-8-sig")
    r = ingest_business(str(d), str(tmp_path / "w"))
    t = r.tables[0]
    assert t.header_row == 3 and t.summary_rows == 1                         # E2, E4
    kinds = {c.name: (c.kind, c.units) for c in t.columns}
    assert kinds["sl"] == ("number", ["bao"])                                # E8
    rows = _typed(r, t.name)
    assert rows[0][3] == 1250000 and rows[1][3] == -450000                   # E9 + VN format
    assert str(rows[0][1]) == "2026-03-05"


def test_cp1258_encoding(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    (d / "kh.csv").write_bytes("Ma KH,Ten\nKH1,Chi Lan\nKH2,Anh Tu\n".encode("cp1258"))
    r = ingest_business(str(d), str(tmp_path / "w"))
    assert r.tables and r.tables[0].n_rows == 2                              # E1


def test_xlsx_hidden_sheet_multiplier_and_side_by_side(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "SP"
    ws.append(["DANH MỤC"])
    ws.append([])
    ws.append(["Mã SP", "Tên SP", "Giá (nghìn)", None, None, "Mã KH", "Tên KH"])
    ws.append(["SP1", "Xi măng", 92, None, None, "KH1", "Chị Lan"])
    ws.append(["SP2", "Cát vàng", 350, None, None, "KH2", "Anh Tú"])
    h = wb.create_sheet("Ẩn")
    h.sheet_state = "hidden"
    h.append(["Mã NCC", "Tên NCC"])
    h.append(["N1", "Hòa Phát"])
    h.append(["N2", "Hà Tiên"])
    wb.save(d / "f.xlsx")
    r = ingest_business(str(d), str(tmp_path / "w"))
    names = sorted(t.name for t in r.tables)
    assert len(names) == 3                                                   # E10 hidden + E11 side by side
    price = next(c for t in r.tables for c in t.columns if c.orig_name == "Giá (nghìn)")
    assert price.unit_multiplier == 1000                                     # E16


def test_two_level_header():
    grid = [["Sản phẩm", "Doanh thu", "", "Số lượng", ""], ["", "Tháng 1", "Tháng 2", "Tháng 1", "Tháng 2"],
            ["Sting", 100, 120, 10, 12], ["Coca", 90, 80, 9, 8]]
    t = grid_to_tables(grid)[0]
    assert t.columns[1] == "Doanh thu_Tháng 1" and t.columns[4] == "Số lượng_Tháng 2"   # E3


def test_split_blocks_and_header_detection():
    grid = [["Báo cáo tháng 3"], [], ["Mã", "Tên", "SL"], ["A1", "x", 1], ["A2", "y", 2], [], ["Ghi chú: số liệu tạm"]]
    blocks = split_blocks(grid)
    assert len(blocks) == 3
    h, n = detect_header(blocks[1][2])
    assert h == 0 and n == 1


def test_sqlite_view_readonly_and_bad_files(tmp_path):
    d = tmp_path / "b"
    d.mkdir()
    p = d / "x.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE t(a TEXT, b INTEGER)")
    c.execute("INSERT INTO t VALUES ('A1', 3), ('A2', 4)")
    c.execute("CREATE VIEW v AS SELECT a, b * 2 AS b2 FROM t")
    c.commit()
    c.close()
    (d / "empty.csv").write_bytes(b"")
    (d / "broken.xlsx").write_bytes(b"not a zip")
    mtime = p.stat().st_mtime
    r = ingest_business(str(d), str(tmp_path / "w"))
    assert {t.name for t in r.tables} >= {"t", "v"} or len(r.tables) >= 2     # E12
    assert any(t.is_view for t in r.tables)
    assert any("empty" in w for w in r.warnings) and any("broken" in w for w in r.warnings)   # E15
    assert r.readme == ""                                                    # E14
    assert p.stat().st_mtime == mtime and not (d / "x.sqlite-journal").exists()
