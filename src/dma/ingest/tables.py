"""Grid -> logical tables: block splitting, header detection, row cleaning (05.1 §4–6)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from dma.ingest.parsers import parse_date, parse_number
from dma.ingest.readers import cell_to_str
from dma.textnorm import fold, nfc, strip_accents

HEADER_LEXICON = {
    "ma", "ten", "ngay", "so", "luong", "sl", "gia", "don", "tien", "thanh", "tong", "khach", "hang", "sp",
    "san", "pham", "loai", "nhom", "danh", "muc", "trang", "thai", "ghi", "chu", "dvt", "ck", "chiet",
    "khau", "vat", "thue", "phi", "id", "name", "date", "qty", "price", "amount", "total", "status",
    "nhan", "vien", "chi", "nhanh", "kenh", "thanh", "toan", "no", "ton", "kho", "nhap", "lo", "hsd",
    "dich", "vu", "stt", "code", "type", "category", "customer", "product", "order", "hd", "kh", "nv",
}
# the whole label must be a total word ("Tổng hợp gia vị" is a product, not a total row)
SUMMARY_RE = re.compile(r"^(tong|cong|tong cong|total|subtotal|sub total|sum|tc|grand total|cong don)"
                        r"( (thang|tuan|ngay|nam|quy|ky|cong|tien|so|ket|cuoi|dau|toan)\b.*| \d{1,2} \d{1,2}( \d{2,4})?)?$")
# a label cell elsewhere in the row must look like a total ("Tổng ngày 01/04"), never a name ("Tống Thắng Long")
TOTAL_EXACT = {"tong", "tong cong", "cong", "total", "grand total", "subtotal", "sub total", "tong so", "tong tien",
               "tong thang", "tong tuan", "tong ngay", "tong nam", "tong quy", "tong ky", "cong thang", "cong tuan",
               "cong ngay", "cong nam", "tong cong thang", "tong cong nam", "tong toan ky", "tong cuoi ky"}
UNIT_RE = [
    (re.compile(r"\b(dvt|don vi tinh|don vi)\s*:?\s*(nghin|ngan)\s*(dong|vnd)?\b"), 1000.0),
    (re.compile(r"\b(dvt|don vi tinh|don vi)\s*:?\s*trieu\s*(dong|vnd)?\b"), 1e6),
    (re.compile(r"\((nghin|ngan|k|nghin dong|ngan dong)\)"), 1000.0),
    (re.compile(r"\((trieu|tr|trieu dong)\)"), 1e6),
]


@dataclass
class RawTable:
    columns: list[str]                 # original header names (unique)
    rows: list[list[str]]              # data cells as strings
    header_row: int                    # 0-based row index in the grid
    start_col: int = 0
    preamble: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary_rows: list[list[str]] = field(default_factory=list)
    dropped_repeated_headers: int = 0
    duplicate_rows: int = 0
    table_multiplier: float = 1.0      # from "ĐVT: nghìn đồng" in the preamble
    group_key: str | None = None       # column whose blanks mark continuation rows of a group (order)
    filled_columns: list[str] = field(default_factory=list)


def _filled(v: Any) -> bool:
    return v is not None and cell_to_str(v).strip() != ""


def split_blocks(grid: list[list[Any]]) -> list[tuple[int, int, list[list[Any]]]]:
    """-> [(row0, col0, block_grid)]; vertical split on blank rows, then horizontal on >=2 blank cols."""
    if not grid:
        return []
    width = max((len(r) for r in grid), default=0)
    g = [list(r) + [""] * (width - len(r)) for r in grid]
    blank_row = [not any(_filled(v) for v in r) for r in g]
    vblocks, cur, start = [], [], 0
    for i, r in enumerate(g):
        if blank_row[i]:
            if cur:
                vblocks.append((start, cur))
                cur = []
            continue
        if not cur:
            start = i
        cur.append(r)
    if cur:
        vblocks.append((start, cur))
    out = []
    for r0, blk in vblocks:
        colfilled = [any(_filled(r[c]) for r in blk) for c in range(width)]
        segs, c, seg_start, gap = [], 0, None, 0
        for c in range(width):
            if colfilled[c]:
                if seg_start is None:
                    seg_start = c
                gap = 0
                last = c
            else:
                gap += 1
                if seg_start is not None and gap >= 2:
                    segs.append((seg_start, last))
                    seg_start = None
        if seg_start is not None:
            segs.append((seg_start, last))
        for c0, c1 in segs:
            sub = [r[c0:c1 + 1] for r in blk]
            out.append((r0, c0, sub))
    return out


def merge_title_blocks(blocks: list[tuple[int, int, list[list[Any]]]]) -> list[tuple[int, int, list[list[Any]], list[str]]]:
    """Small blocks (< 2 rows or < 2 filled cols) are preambles of the next real block."""
    out, pending = [], []
    for r0, c0, b in blocks:
        filled_cols = sum(any(_filled(r[c]) for r in b) for c in range(len(b[0]) if b else 0))
        if len(b) < 2 or filled_cols < 2:
            pending += [" ".join(cell_to_str(v).strip() for v in r if _filled(v)) for r in b]
            continue
        out.append((r0, c0, b, pending))
        pending = []
    if pending and out:
        out[-1][3].extend(pending)
    return out


def _is_numlike(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float, datetime, date)):
        return True
    s = cell_to_str(v).strip()
    return bool(s) and (parse_number(s, allow_unit=False) is not None or parse_date(s) is not None)


def header_score(rows: list[list[Any]], i: int) -> float:
    r = rows[i]
    width = len(r)
    cells = [v for v in r if _filled(v)]
    if len(cells) < 2:
        return -9.0
    texty = sum(not _is_numlike(v) for v in cells) / len(cells)
    filled = len(cells) / max(1, width)
    nxt = rows[i + 1:i + 6]
    if nxt:
        # rows below a header fill a SUBSET of its columns (optional columns may be blank everywhere)
        hdr_pos = {c for c in range(width) if _filled(r[c])}
        ok = 0
        for rr in nxt:
            pos = {c for c in range(min(width, len(rr))) if _filled(rr[c])}
            ok += len(pos) >= 2 and pos <= hdr_pos
        consistency = ok / len(nxt)
    else:
        consistency = 0.0
    shift = 0.0
    if nxt:
        cols = [c for c in range(width) if _filled(r[c]) and not _is_numlike(r[c])]
        if cols:
            shift = sum(any(_is_numlike(rr[c]) for rr in nxt if c < len(rr)) for c in cols) / len(cols)
    toks = [t for v in cells for t in fold(cell_to_str(v)).split()]
    lex = min(1.0, sum(t in HEADER_LEXICON for t in toks) / max(1, len(cells)))
    is_title = 1.0 if len(cells) == 1 and len(cell_to_str(cells[0])) > 15 else 0.0
    return 2.0 * texty + 1.5 * filled + 2.0 * consistency + 1.0 * shift + 1.0 * lex - 2.0 * is_title


def detect_header(rows: list[list[Any]]) -> tuple[int, int]:
    """-> (header_index, n_header_rows in {1, 2})."""
    n = min(20, len(rows))
    if n == 0:
        return 0, 1
    scores = [header_score(rows, i) for i in range(n)]
    h = max(range(n), key=lambda i: (scores[i], -i))
    if h > 0:   # the best row may be the LOWER level of a 2-level header: look one row up
        up, r = rows[h - 1], rows[h]
        cells_up = [v for v in up if _filled(v)]
        if len(cells_up) >= 2 and all(not _is_numlike(v) for v in cells_up) and \
                any(not _filled(up[c]) and _filled(r[c]) for c in range(min(len(up), len(r)))) and \
                sum(_filled(v) for v in r) > len(cells_up):
            return h - 1, 2
    two = False
    if h + 2 < len(rows):
        r1, r2 = rows[h], rows[h + 1]
        cells2 = [v for v in r2 if _filled(v)]
        texty2 = sum(not _is_numlike(v) for v in cells2) / max(1, len(cells2))
        repeats = any(_filled(r1[c]) and c + 1 < len(r1) and cell_to_str(r1[c]) == cell_to_str(r1[c + 1])
                      for c in range(len(r1) - 1))
        gaps = any(not _filled(r1[c]) and _filled(r2[c]) for c in range(min(len(r1), len(r2))))
        below = rows[h + 2:h + 6]
        data_numeric = any(_is_numlike(v) for rr in below for v in rr)
        if texty2 >= 0.9 and len(cells2) >= 2 and (repeats or gaps) and data_numeric:
            two = True
    return h, 2 if two else 1


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        k = n
        if k in seen:
            seen[k] += 1
            k = f"{n}_{seen[n]}"
        else:
            seen[k] = 1
        out.append(k)
    return out


def detect_multiplier(texts: list[str]) -> float:
    for t in texts:
        plain = re.sub(r"\s+", " ", strip_accents(nfc(str(t)).lower()))   # accents gone, "(" kept
        for rx, mult in UNIT_RE:
            if rx.search(fold(t)) or rx.search(plain):
                return mult
    return 1.0


def build_table(block: list[list[Any]], preamble: list[str], r0: int = 0, c0: int = 0) -> RawTable | None:
    h, nh = detect_header(block)
    width = len(block[0]) if block else 0
    top = block[h]
    if nh == 2:
        sub = block[h + 1]
        names, last = [], ""
        for c in range(width):
            a = cell_to_str(top[c]).strip() if c < len(top) else ""
            b = cell_to_str(sub[c]).strip() if c < len(sub) else ""
            if a:
                last = a
            else:
                a = last if b else ""
            names.append(f"{a}_{b}" if a and b and a != b else (a or b))
    else:
        names = [cell_to_str(top[c]).strip() if c < len(top) else "" for c in range(width)]
    names = [re.sub(r"\s+", " ", n) or f"col_{i}" for i, n in enumerate(names)]
    names = _unique(names)
    pre = list(preamble) + [" ".join(cell_to_str(v).strip() for v in r if _filled(v)) for r in block[:h]]
    t = RawTable(names, [], r0 + h, c0, [p for p in pre if p])
    header_key = [fold(cell_to_str(v)) for v in top]
    seen_hash: set[tuple] = set()
    data = block[h + nh:]
    first_filled_col = [any(_filled(r[c]) for r in data[:50]) for c in range(width)]
    for r in data:
        cells = [cell_to_str(v).strip() if c < len(r) else "" for c, v in enumerate(r[:width])]
        cells += [""] * (width - len(cells))
        nfill = sum(bool(c) for c in cells)
        if nfill == 0:
            continue
        if [fold(c) for c in cells] == header_key:
            t.dropped_repeated_headers += 1
            continue
        lead = next((fold(c) for c in cells[:3] if c and not _is_numlike(c)), "")
        labelled = any(c and not _is_numlike(c) and SUMMARY_RE.match(fold(c)) and fold(c) not in header_key for c in cells)
        # a total label with at most one other cell and a blank key column is a footer ("TỔNG THÁNG | 227.690.000đ")
        if (SUMMARY_RE.match(lead) and (nfill < width or not cells[0])) or \
                (labelled and any(re.search(r"\d", fold(c)) or fold(c) in TOTAL_EXACT
                                  for c in cells if c and SUMMARY_RE.match(fold(c)))) or \
                (labelled and nfill <= 2 and not cells[0] and width >= 4):
            t.summary_rows.append(cells)
            continue
        only = next(c for c in cells if c) if nfill == 1 else ""
        if nfill == 1 and sum(first_filled_col) > 2 and (
                len(only) > 40 or re.match(r"[^\d:/]{2,30}:\s*\S", only)):     # "Ghi chú: tiền tính bằng đồng"
            t.notes.append(next(c for c in cells if c))
            continue
        key = tuple(cells)
        if key in seen_hash:
            t.duplicate_rows += 1
        seen_hash.add(key)
        t.rows.append(cells)
    if not t.rows:
        return None
    t.table_multiplier = detect_multiplier(t.preamble + t.notes)
    fill_groups(t)
    return t


def fill_groups(t: RawTable, min_cover: float = 0.98) -> None:
    """Rulebook §8: an empty cell in the following rows of a group repeats the group's first row.

    Detected from the blank pattern only (no column names): the key K is a column that is blank on
    some non-empty rows, never on the first row; header columns are those blank on (almost) every
    row where K is blank. Those cells are forward-filled from the group's first row."""
    n, w = len(t.rows), len(t.columns)
    if n < 3:
        return
    blanks = [{i for i, r in enumerate(t.rows) if not r[c]} for c in range(w)]
    best = None
    for k in range(w):
        bk = blanks[k]
        if not bk or 0 in bk or len(bk) >= 0.95 * n:
            continue
        filled_vals = [t.rows[i][k] for i in range(n) if i not in bk]
        if len(set(filled_vals)) < 0.5 * len(filled_vals):      # a group key is (nearly) unique per group
            continue
        hdr = [c for c in range(w) if c != k and len(blanks[c] & bk) >= min_cover * len(bk)]
        line_cols = [c for c in range(w) if c != k and c not in hdr and len(blanks[c] & bk) <= 0.2 * len(bk)]
        if not line_cols or not hdr:                           # continuation rows carry line data, not header data
            continue
        # a real group header (customer, date, status) is written on the group's first row; columns that are
        # mostly blank everywhere (a note) do not make the blanks of an optional column (IMEI) a group pattern
        keyed = n - len(bk)
        if not any(sum(1 for i in range(n) if i not in bk and t.rows[i][c]) >= 0.9 * keyed for c in hdr):
            continue
        cand = (len(hdr), len(bk), -k, k, hdr)
        best = max(best, cand) if best else cand
    if best is None:
        return
    _, _, _, k, hdr = best
    cols = [k] + hdr
    last = list(t.rows[0])
    for r in t.rows:
        if not r[k]:
            for c in cols:
                if not r[c]:
                    r[c] = last[c]
        else:
            last = list(r)
    t.group_key = t.columns[k]
    t.filled_columns = [t.columns[c] for c in cols]


def grid_to_tables(grid: list[list[Any]]) -> list[RawTable]:
    out = []
    for r0, c0, blk, pre in merge_title_blocks(split_blocks(grid)):
        t = build_table(blk, pre, r0, c0)
        if t is not None and len(t.columns) >= 2:
            out.append(t)
    return out


def is_document_sheet(grid: list[list[Any]]) -> bool:
    """A notes/instructions sheet: mostly long free text, little structure."""
    cells = [cell_to_str(v) for r in grid for v in r if _filled(v)]
    if not cells:
        return False
    long_text = sum(len(c) > 30 and not _is_numlike(c) for c in cells)
    widths = [sum(_filled(v) for v in r) for r in grid if any(_filled(v) for v in r)]
    return long_text / len(cells) >= 0.6 and (sum(w <= 1 for w in widths) / max(1, len(widths))) >= 0.6
