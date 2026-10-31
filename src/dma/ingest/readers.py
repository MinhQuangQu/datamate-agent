"""File discovery + raw readers (05.1 §2–3). Readers return GRIDS of cells (no type
guessing by pandas); SQLite tables come back as DataFrames with declared types and FKs.
Nothing here ever writes into the data directory.
"""
from __future__ import annotations

import csv
import io
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

SUPPORTED = {".csv": "csv", ".tsv": "csv", ".txt": "csv", ".xlsx": "xlsx", ".xlsm": "xlsx",
             ".xls": "xls", ".sqlite": "sqlite", ".sqlite3": "sqlite", ".db": "sqlite"}
ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "utf-16")
DELIMS = [",", ";", "\t", "|"]


@dataclass
class FileRef:
    path: str
    relpath: str
    kind: str          # csv | xlsx | xls | sqlite | readme | other


@dataclass
class Sheet:
    name: str
    grid: list[list[Any]]
    hidden: bool = False


@dataclass
class SqliteTable:
    name: str
    columns: list[str]
    rows: list[tuple]
    declared: dict[str, str] = field(default_factory=dict)
    fks: list[tuple[str, str, str]] = field(default_factory=list)    # (from_col, to_table, to_col)
    is_view: bool = False


def discover_files(biz_dir: str) -> tuple[list[FileRef], list[str]]:
    out, warnings = [], []
    for root, dirs, files in os.walk(biz_dir):
        dirs.sort()
        for fn in sorted(files):
            if fn.startswith(("~$", ".")):
                continue
            p = os.path.join(root, fn)
            rel = os.path.relpath(p, biz_dir)
            ext = os.path.splitext(fn)[1].lower()
            if fn.lower() == "readme.md":
                out.append(FileRef(p, rel, "readme"))
            elif ext in SUPPORTED:
                out.append(FileRef(p, rel, SUPPORTED[ext]))
            elif ext in (".md", ".txt"):
                out.append(FileRef(p, rel, "doc"))
            else:
                warnings.append(f"unsupported file skipped: {rel}")
    return sorted(out, key=lambda f: f.relpath), warnings


def decode_bytes(raw: bytes) -> tuple[str, str]:
    for enc in ENCODINGS:
        try:
            text = raw.decode(enc)
            if enc == "utf-16" and "\x00" in text[:200]:
                continue
            return text, enc
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best is not None:
            return str(best), f"detected:{best.encoding}"
    except ImportError:
        pass
    return raw.decode("latin-1", errors="replace"), "latin-1"


def _width_stability(text: str, delim: str, n: int = 200) -> tuple[float, int]:
    rows = []
    try:
        for i, r in enumerate(csv.reader(io.StringIO(text[:200_000]), delimiter=delim)):
            if i >= n:
                break
            if any(c.strip() for c in r):
                rows.append(len(r))
    except csv.Error:
        return 0.0, 0
    if not rows:
        return 0.0, 0
    mode = max(set(rows), key=lambda w: (rows.count(w), w))
    return rows.count(mode) / len(rows), mode


def sniff_delimiter(text: str) -> str:
    try:
        d = csv.Sniffer().sniff(text[:8192], delimiters="".join(DELIMS)).delimiter
    except csv.Error:
        d = None
    best, best_key = ",", (-1.0, 0)
    for cand in DELIMS:
        stab, width = _width_stability(text, cand)
        key = (stab if width >= 2 else 0.0, width)
        if key > best_key:
            best, best_key = cand, key
    if d is not None:
        stab, width = _width_stability(text, d)
        if width >= 2 and stab >= best_key[0] - 1e-9:
            return d
    return best


def read_csv_grid(path: str) -> tuple[list[list[str]], dict]:
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.strip():
        return [], {"encoding": None, "delimiter": None, "empty": True}
    text, enc = decode_bytes(raw)
    delim = sniff_delimiter(text)
    grid = [list(r) for r in csv.reader(io.StringIO(text), delimiter=delim)]
    return grid, {"encoding": enc, "delimiter": delim}


def _cell(v: Any) -> Any:
    if isinstance(v, str):
        return v
    if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
        return int(v)
    return v


def read_xlsx_sheets(path: str, merged_fill_max_bytes: int = 3_000_000) -> tuple[list[Sheet], list[str]]:
    warnings: list[str] = []
    sheets: list[Sheet] = []
    try:
        from python_calamine import CalamineWorkbook
        wb = CalamineWorkbook.from_path(path)
        meta = {}
        try:
            for sm in wb.sheets_metadata:
                meta[sm.name] = str(getattr(sm, "visible", "")).lower()
        except Exception:  # noqa: BLE001
            pass
        for name in wb.sheet_names:
            try:
                rows = wb.get_sheet_by_name(name).to_python(skip_empty_area=False)
            except Exception as e:  # noqa: BLE001
                warnings.append(f"sheet {name}: {e}")
                continue
            grid = [[_cell(v) for v in r] for r in rows]
            sheets.append(Sheet(name, grid, hidden="hidden" in meta.get(name, "")))
    except Exception as e:  # noqa: BLE001
        warnings.append(f"calamine failed ({e}); trying openpyxl")
        sheets = _read_openpyxl(path, warnings, values_only=True)
    if sheets and os.path.getsize(path) <= merged_fill_max_bytes:
        _fill_merged(path, sheets, warnings)
    return sheets, warnings


def _read_openpyxl(path: str, warnings: list[str], values_only: bool = True) -> list[Sheet]:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"openpyxl failed: {e}")
        return []
    out = []
    for ws in wb.worksheets:
        grid = [[_cell(v) if v is not None else "" for v in r] for r in ws.iter_rows(values_only=values_only)]
        out.append(Sheet(ws.title, grid, getattr(ws, "sheet_state", "visible") != "visible"))
    return out


def _fill_merged(path: str, sheets: list[Sheet], warnings: list[str]) -> None:
    """Forward-fill merged ranges (header area only) so 2-level headers get names."""
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=False, data_only=True)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"merged-cell scan skipped: {e}")
        return
    by_name = {s.name: s for s in sheets}
    for ws in wb.worksheets:
        s = by_name.get(ws.title)
        if s is None:
            continue
        s.hidden = s.hidden or ws.sheet_state != "visible"
        for rng in ws.merged_cells.ranges:
            r0, c0, r1, c1 = rng.min_row - 1, rng.min_col - 1, rng.max_row - 1, rng.max_col - 1
            if r0 > 40 or r0 >= len(s.grid):
                continue
            v = s.grid[r0][c0] if c0 < len(s.grid[r0]) else ""
            for r in range(r0, min(r1, len(s.grid) - 1) + 1):
                row = s.grid[r]
                while len(row) <= c1:
                    row.append("")
                for c in range(c0, c1 + 1):
                    if row[c] in ("", None):
                        row[c] = v


def read_xls_sheets(path: str) -> tuple[list[Sheet], list[str]]:
    try:
        from python_calamine import CalamineWorkbook   # calamine reads legacy .xls too
        wb = CalamineWorkbook.from_path(path)
        return [Sheet(n, [[_cell(v) for v in r] for r in wb.get_sheet_by_name(n).to_python(skip_empty_area=False)])
                for n in wb.sheet_names], []
    except Exception as e:  # noqa: BLE001
        return [], [f".xls unreadable: {e}"]


def read_sqlite(path: str) -> tuple[list[SqliteTable], list[str]]:
    warnings: list[str] = []
    out: list[SqliteTable] = []
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)
    except sqlite3.Error as e:
        return [], [f"sqlite open failed: {e}"]
    try:
        objs = con.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                           "AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()
        for name, typ in objs:
            try:
                qn = '"' + name.replace('"', '""') + '"'
                info = con.execute(f"PRAGMA table_info({qn})").fetchall()
                cols = [r[1] for r in info]
                declared = {r[1]: (r[2] or "").upper() for r in info}
                fks = []
                if typ == "table":
                    for r in con.execute(f"PRAGMA foreign_key_list({qn})").fetchall():
                        fks.append((r[3], r[2], r[4]))
                cur = con.execute(f"SELECT * FROM {qn}")
                if not cols:
                    cols = [d[0] for d in cur.description]
                rows = cur.fetchall()
                out.append(SqliteTable(name, cols, rows, declared, fks, typ == "view"))
            except sqlite3.Error as e:
                warnings.append(f"sqlite {name}: {e}")
    finally:
        con.close()
    return out, warnings


def read_text(path: str, max_bytes: int = 200_000) -> str:
    try:
        with open(path, "rb") as f:
            raw = f.read(max_bytes)
        return decode_bytes(raw)[0].strip()
    except OSError:
        return ""


def cell_to_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"
    if isinstance(v, float):
        if v != v:   # NaN
            return ""
        return str(int(v)) if v.is_integer() and abs(v) < 1e15 else repr(v)
    if isinstance(v, datetime):
        return v.isoformat(sep=" ") if (v.hour or v.minute or v.second) else v.date().isoformat()
    if isinstance(v, date):
        return v.isoformat()
    return str(v)
