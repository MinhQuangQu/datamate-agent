"""ingest_business(): every file of one business -> <work_dir>/db.duckdb (05.1 §7–8, §13).

Schemas: raw.* (VARCHAR, original names, _row_id), typed.* (snake_case names, inferred
types), meta.tables / meta.columns (provenance, formats, unit multipliers). Never raises:
per-file problems become warnings. typed.* keeps numbers AS WRITTEN; unit multipliers
("ĐVT: nghìn đồng") are recorded in meta and applied by the semantic views.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from dma.ingest.parsers import (detect_date_order, detect_number_format, infer_column, parse_bool,
                                parse_date, parse_number)
from dma.ingest.readers import (cell_to_str, discover_files, read_csv_grid, read_sqlite, read_text,
                                read_xls_sheets, read_xlsx_sheets)
from dma.ingest.tables import RawTable, detect_multiplier, grid_to_tables, is_document_sheet
from dma.textnorm import fold

MONEY_HINT = ("tien", "gia", "doanh thu", "amount", "price", "total", "chi phi", "cong no", "no ", "luong",
              "phi", "von", "thanh toan", "giam", "chiet khau", "ck", "vat", "thue", "voucher")
SAMPLE = 5000


@dataclass
class ColumnMeta:
    orig_name: str
    name: str
    kind: str
    sql_type: str
    number_format: str | None = None
    date_order: str | None = None
    units: list[str] = field(default_factory=list)
    unit_multiplier: float = 1.0
    parse_fail: int = 0
    null_rate: float = 0.0
    declared_type: str | None = None


@dataclass
class LoadedTable:
    name: str
    source: str
    n_rows: int
    columns: list[ColumnMeta]
    header_row: int | None = None
    preamble: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_view: bool = False
    summary_rows: int = 0
    duplicate_rows: int = 0
    declared_fks: list[tuple[str, str, str]] = field(default_factory=list)   # (col, table, col) in typed names


@dataclass
class IngestResult:
    business_dir: str
    db_path: str
    tables: list[LoadedTable] = field(default_factory=list)
    files: list[dict] = field(default_factory=list)
    readme: str = ""
    docs: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    build_s: float = 0.0
    ok: bool = True

    def to_json(self) -> dict:
        return asdict(self)


def sql_ident(s: str, maxlen: int = 60) -> str:
    x = re.sub(r"[^a-z0-9]+", "_", fold(s)).strip("_")
    if not x:
        x = "t"
    if x[0].isdigit():
        x = "c_" + x
    return x[:maxlen].rstrip("_")


def q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _uniq(name: str, used: set[str]) -> str:
    base, k, i = name, name, 2
    while k in used:
        k = f"{base}_{i}"
        i += 1
    used.add(k)
    return k


def _header_multiplier(orig: str) -> float:
    return detect_multiplier([orig]) if "(" in orig else 1.0


def _memo_parse(values: list[Any], fn) -> list[Any]:
    cache: dict = {}
    out = []
    for v in values:
        key = v if isinstance(v, (str, int, float, bool)) or v is None else repr(v)
        if key not in cache:
            cache[key] = fn(v)
        out.append(cache[key])
    return out


def _empty(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def type_column(values: list[Any], name: str) -> tuple[ColumnMeta, list[Any]]:
    """Decide kind/format on a sample, then parse the full column (memoised by value)."""
    nonempty = [v for v in values if not _empty(v)]
    sample = nonempty[:SAMPLE] if len(nonempty) <= SAMPLE else \
        [nonempty[int(i * len(nonempty) / SAMPLE)] for i in range(SAMPLE)]
    ci = infer_column(sample, name)
    kind = ci.kind
    cm = ColumnMeta(orig_name=name, name="", kind=kind, sql_type="VARCHAR",
                    null_rate=round(1 - len(nonempty) / max(1, len(values)), 4))
    if kind in ("date", "datetime"):
        order = ci.date_order or detect_date_order(sample)
        date_name = any(h in fold(name) for h in ("ngay", "date", "thoi gian", "time", "han", "hsd", "exp", "nsx"))
        parsed = _memo_parse(values, lambda v: None if _empty(v) else parse_date(v, order, allow_excel_serial=date_name))
        cm.date_order = order
        if kind == "date":
            parsed = [p.date() if isinstance(p, datetime) else p for p in parsed]
            cm.sql_type = "DATE"
        else:
            cm.sql_type = "TIMESTAMP"
    elif kind in ("number", "percent"):
        fmt = ci.number_format if ci.number_format is not None else detect_number_format(sample, name)
        parsed = _memo_parse(values, lambda v: None if _empty(v) else (lambda p: None if p is None else p.value)(parse_number(v, fmt)))
        cm.number_format = None if fmt is None else ("VN" if fmt.decimal == "," else "US")
        cm.units = ci.units
        nums = [p for p in parsed if p is not None]
        integral = kind == "number" and all(float(p).is_integer() and abs(p) < 9e15 for p in nums)
        if integral:
            parsed = [None if p is None else int(p) for p in parsed]
            cm.sql_type = "BIGINT"
        else:
            cm.sql_type = "DOUBLE"
    elif kind == "bool":
        flag = True
        parsed = _memo_parse(values, lambda v: None if _empty(v) else parse_bool(v, allow_01=flag))
        cm.sql_type = "BOOLEAN"
    elif kind == "id":
        parsed = [None if _empty(v) else cell_to_str(v).strip() for v in values]
    else:
        parsed = [None if _empty(v) else re.sub(r"[ \t]+$", "", cell_to_str(v).strip()) for v in values]
    cm.parse_fail = sum(1 for v, p in zip(values, parsed) if not _empty(v) and p is None)
    return cm, parsed


def _register(con, df_name: str, cols: list[str], data: list[list[Any]], types: list[str]) -> None:
    import pandas as pd
    series = {}
    for c, vals, t in zip(cols, data, types):
        if t == "BIGINT":
            series[c] = pd.array(vals, dtype="Int64")
        elif t == "DOUBLE":
            series[c] = pd.array([None if v is None else float(v) for v in vals], dtype="Float64")
        elif t == "BOOLEAN":
            series[c] = pd.array(vals, dtype="boolean")
        elif t in ("DATE", "TIMESTAMP"):
            series[c] = pd.to_datetime(pd.Series(vals, dtype="object"), errors="coerce")
        else:
            series[c] = pd.Series(vals, dtype="object")
    con.register(df_name, pd.DataFrame(series))


_DAY_MONTH = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})\s*$")
_YEAR_IN_NAME = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _add_sheet_year(values: list[Any], name: str, source: str) -> list[Any]:
    """Rulebook §8: a date written without a year belongs to the year of its sheet ("T03_2025")."""
    sheet = source.split("!")[1].split("@")[0] if "!" in source else ""
    m = _YEAR_IN_NAME.search(sheet) or None
    if not m or not any(h in fold(name) for h in ("ngay", "date", "thoi gian")):
        return values
    filled = [v for v in values if isinstance(v, str) and v.strip()]
    if not filled or sum(bool(_DAY_MONTH.match(v)) for v in filled) < 0.05 * len(filled):
        return values
    y = m.group(1)
    return [f"{v.strip()}/{y}" if isinstance(v, str) and _DAY_MONTH.match(v) else v for v in values]


def _load_table(con, name: str, source: str, columns: list[str], rows: list[list[Any]], used_cols_seed=None,
                table_mult: float = 1.0, declared: dict[str, str] | None = None) -> LoadedTable:
    ncol = len(columns)
    colvals = [[(r[j] if j < len(r) else None) for r in rows] for j in range(ncol)]
    colvals = [_add_sheet_year(cv, columns[j], source) for j, cv in enumerate(colvals)]
    metas, parsed_cols = [], []
    used: set[str] = set()
    for j, orig in enumerate(columns):
        cm, parsed = type_column(colvals[j], orig)
        if declared and declared.get(orig, "").startswith(("INT", "REAL", "NUM", "FLOAT", "DOUB")) and cm.kind in ("id", "string"):
            cm.declared_type = declared.get(orig)
        elif declared:
            cm.declared_type = declared.get(orig)
        cm.name = _uniq(sql_ident(orig) or f"col_{j}", used)
        hm = _header_multiplier(orig)
        if hm != 1.0:
            cm.unit_multiplier = hm
        elif table_mult != 1.0 and cm.kind == "number" and any(h in fold(orig) + " " for h in MONEY_HINT):
            cm.unit_multiplier = table_mult
        metas.append(cm)
        parsed_cols.append(parsed)
    # raw: VARCHAR, original names
    raw_cols = ["_row_id"] + _raw_names(columns)
    raw_data = [list(range(len(rows)))] + [[cell_to_str(v) if not _empty(v) else None for v in cv] for cv in colvals]
    _register(con, "_raw_df", raw_cols, raw_data, ["BIGINT"] + ["VARCHAR"] * ncol)
    con.execute(f"CREATE OR REPLACE TABLE raw.{q(name)} AS SELECT * FROM _raw_df")
    con.unregister("_raw_df")
    tcols = [m.name for m in metas]
    _register(con, "_typed_df", tcols, parsed_cols, [m.sql_type for m in metas])
    sel = ", ".join(f"CAST({q(m.name)} AS {m.sql_type}) AS {q(m.name)}" for m in metas)
    con.execute(f"CREATE OR REPLACE TABLE typed.{q(name)} AS SELECT {sel} FROM _typed_df")
    con.unregister("_typed_df")
    lt = LoadedTable(name=name, source=source, n_rows=len(rows), columns=metas)
    for m in metas:
        nonnull = len(rows) * (1 - m.null_rate)
        if nonnull and m.parse_fail / nonnull > 0.005:
            lt.warnings.append(f"{m.name}: {m.parse_fail} cells not parsed as {m.kind} (NULL in typed)")
    return lt


def _raw_names(columns: list[str]) -> list[str]:
    used: set[str] = {"_row_id"}
    return [_uniq(c or f"col_{i}", used) for i, c in enumerate(columns)]


def ingest_business(biz_dir: str, work_dir: str) -> IngestResult:
    import duckdb
    t0 = time.monotonic()
    os.makedirs(work_dir, exist_ok=True)
    db_path = os.path.join(work_dir, "db.duckdb")
    for p in (db_path, db_path + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    res = IngestResult(business_dir=biz_dir, db_path=db_path)
    try:
        con = duckdb.connect(db_path)
    except Exception as e:  # noqa: BLE001
        res.ok, res.warnings = False, [f"duckdb open failed: {e}"]
        return res
    try:
        tmp = os.path.join(os.path.dirname(os.path.dirname(work_dir)) or "/tmp", "tmp")
        os.makedirs(tmp, exist_ok=True)
        con.execute("SET threads = 2")
        con.execute(f"SET temp_directory = '{tmp}'")
        for s in ("raw", "typed", "meta", "sem"):
            con.execute(f"CREATE SCHEMA IF NOT EXISTS {s}")
        files, warns = discover_files(biz_dir)
        res.warnings += warns
        used_tables: set[str] = set()
        stems = [os.path.splitext(f.relpath)[0] for f in files if f.kind not in ("readme", "doc")]
        for f in files:
            entry = {"file": f.relpath, "kind": f.kind, "bytes": _size(f.path), "tables": []}
            res.files.append(entry)
            try:
                if f.kind == "readme":
                    res.readme = read_text(f.path)
                    continue
                if f.kind == "doc":
                    txt = read_text(f.path, 20_000)
                    if txt:
                        res.docs.append(f"[{f.relpath}]\n{txt}")
                    continue
                stem = sql_ident(os.path.splitext(f.relpath)[0])
                if entry["bytes"] == 0:
                    res.warnings.append(f"{f.relpath}: empty file skipped")
                    continue
                if f.kind == "csv":
                    grid, meta = read_csv_grid(f.path)
                    entry.update(meta)
                    tabs = grid_to_tables(grid)
                    for i, t in enumerate(tabs):
                        nm = _uniq(stem if len(tabs) == 1 else f"{stem}__{i}", used_tables)
                        entry["tables"].append(nm)
                        res.tables.append(_from_raw(con, nm, f"{f.relpath}@R{t.header_row + 1}", t))
                elif f.kind in ("xlsx", "xls"):
                    sheets, w = (read_xlsx_sheets if f.kind == "xlsx" else read_xls_sheets)(f.path)
                    res.warnings += [f"{f.relpath}: {x}" for x in w]
                    for sh in sheets:
                        if not any(any(cell_to_str(v).strip() for v in r) for r in sh.grid):
                            continue
                        if is_document_sheet(sh.grid):
                            txt = "\n".join(" ".join(cell_to_str(v) for v in r if cell_to_str(v).strip()) for r in sh.grid)
                            res.docs.append(f"[{f.relpath}!{sh.name}]\n{txt[:6000]}")
                            continue
                        tabs = grid_to_tables(sh.grid)
                        base = stem if len(sheets) == 1 else f"{stem}__{sql_ident(sh.name, 30)}"
                        for i, t in enumerate(tabs):
                            nm = _uniq(base if len(tabs) == 1 else f"{base}__{i}", used_tables)
                            entry["tables"].append(nm)
                            lt = _from_raw(con, nm, f"{f.relpath}!{sh.name}@R{t.header_row + 1}", t)
                            if sh.hidden:
                                lt.warnings.append("from a hidden sheet")
                            res.tables.append(lt)
                elif f.kind == "sqlite":
                    tabs, w = read_sqlite(f.path)
                    res.warnings += [f"{f.relpath}: {x}" for x in w]
                    multi = len([s for s in stems]) > 1
                    name_map: dict[str, str] = {}
                    for st in tabs:
                        nm = _uniq(sql_ident(f"{stem}__{st.name}" if multi else st.name), used_tables)
                        name_map[st.name] = nm
                        entry["tables"].append(nm)
                        rows = [list(r) for r in st.rows]
                        if not rows:
                            res.warnings.append(f"{f.relpath}:{st.name}: no rows")
                        lt = _load_table(con, nm, f"{f.relpath}:{st.name}", st.columns, rows, declared=st.declared)
                        lt.is_view = st.is_view
                        lt._fks_raw = st.fks  # type: ignore[attr-defined]
                        res.tables.append(lt)
                    for lt in res.tables:
                        raw_fks = getattr(lt, "_fks_raw", None)
                        if raw_fks:
                            colmap = {c.orig_name: c.name for c in lt.columns}
                            for fc, tt, tc in raw_fks:
                                tgt = next((x for x in res.tables if x.name == name_map.get(tt)), None)
                                tcol = None
                                if tgt is not None:
                                    tcol = next((c.name for c in tgt.columns if c.orig_name == tc), None) if tc else None
                                if tgt is not None and colmap.get(fc):
                                    lt.declared_fks.append((colmap[fc], tgt.name, tcol or ""))
                            del lt._fks_raw  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001 - one bad file never kills the business
                res.warnings.append(f"{f.relpath}: {type(e).__name__}: {e}")
                res.warnings.append(traceback.format_exc(limit=2)[-600:])
        _write_meta(con, res)
        con.execute("CREATE OR REPLACE MACRO meta.fold_vn(x) AS trim(regexp_replace(replace(replace(strip_accents("
                    "lower(CAST(x AS VARCHAR))), 'đ', 'd'), 'Đ', 'd'), '\\s+', ' ', 'g'))")
        con.execute("CHECKPOINT")
    except Exception as e:  # noqa: BLE001
        res.ok = False
        res.warnings.append(f"ingest failed: {type(e).__name__}: {e}")
    finally:
        con.close()
    res.build_s = round(time.monotonic() - t0, 3)
    with open(os.path.join(work_dir, "ingest.json"), "w", encoding="utf-8") as f:
        json.dump(res.to_json(), f, ensure_ascii=False, indent=1, default=str)
    return res


def _from_raw(con, name: str, source: str, t: RawTable) -> LoadedTable:
    lt = _load_table(con, name, source, t.columns, t.rows, table_mult=t.table_multiplier)
    lt.header_row = t.header_row
    lt.preamble = t.preamble[:10]
    lt.notes = t.notes[:10]
    lt.summary_rows = len(t.summary_rows)
    lt.duplicate_rows = t.duplicate_rows
    if t.group_key:
        lt.notes.append(f"grouped rows: blank {', '.join(t.filled_columns)} filled from the first row of each "
                        f"'{t.group_key}' group")
    if t.dropped_repeated_headers:
        lt.warnings.append(f"{t.dropped_repeated_headers} repeated header rows dropped")
    return lt


def _size(p: str) -> int:
    try:
        return os.path.getsize(p)
    except OSError:
        return -1


def _write_meta(con, res: IngestResult) -> None:
    con.execute("CREATE OR REPLACE TABLE meta.tables(name VARCHAR, source VARCHAR, n_rows BIGINT, header_row INTEGER, "
                "is_view BOOLEAN, preamble VARCHAR, warnings VARCHAR)")
    con.execute("CREATE OR REPLACE TABLE meta.columns(table_name VARCHAR, name VARCHAR, orig_name VARCHAR, kind VARCHAR, "
                "sql_type VARCHAR, number_format VARCHAR, date_order VARCHAR, units VARCHAR, unit_multiplier DOUBLE, "
                "parse_fail BIGINT, null_rate DOUBLE)")
    for t in res.tables:
        con.execute("INSERT INTO meta.tables VALUES (?,?,?,?,?,?,?)",
                    [t.name, t.source, t.n_rows, t.header_row, t.is_view, " | ".join(t.preamble), " | ".join(t.warnings)])
        for c in t.columns:
            con.execute("INSERT INTO meta.columns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        [t.name, c.name, c.orig_name, c.kind, c.sql_type, c.number_format, c.date_order,
                         ",".join(c.units), c.unit_multiplier, c.parse_fail, c.null_rate])


def ingest_result_from_json(d: dict) -> IngestResult:
    tables = []
    for t in d.get("tables", []):
        cols = [ColumnMeta(**c) for c in t.pop("columns")]
        t["declared_fks"] = [tuple(x) for x in t.get("declared_fks", [])]
        tables.append(LoadedTable(columns=cols, **t))
    d = dict(d)
    d["tables"] = tables
    return IngestResult(**d)


_ = (date,)
