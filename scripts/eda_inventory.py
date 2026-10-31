"""Day-1 EDA #1 — inventory of a split (train / public_test): files, formats, tables.

    python scripts/eda_inventory.py /path/to/train --out reports/inventory

Writes inventory.md (human) + inventory.json (machine). Read-only on the split.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.ingest.parsers import infer_column  # noqa: E402

ENCODINGS = ("utf-8-sig", "utf-8", "cp1258", "utf-16", "latin-1")


def sniff_csv(path: str, max_rows: int = 3000) -> dict:
    raw = open(path, "rb").read()
    enc = next((e for e in ENCODINGS if _decodes(raw, e)), "latin-1")
    text = raw.decode(enc, errors="replace")
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    hdr = guess_header(rows[:30])
    header = rows[hdr] if rows else []
    body = rows[hdr + 1: hdr + 1 + max_rows]
    return {"encoding": enc, "delimiter": delim, "header_row": hdr, "n_rows": max(0, len(rows) - hdr - 1),
            "columns": profile_columns(header, body), "preamble": rows[:hdr]}


def _decodes(raw: bytes, enc: str) -> bool:
    try:
        raw.decode(enc)
        return True
    except UnicodeDecodeError:
        return False


def guess_header(rows: list[list]) -> int:
    """First row that is mostly non-empty text and is followed by >=2 rows of equal width."""
    best = 0
    for i, r in enumerate(rows[:20]):
        cells = [str(c).strip() for c in r]
        filled = [c for c in cells if c]
        if len(filled) < 2:
            continue
        texty = sum(not _is_num(c) for c in filled) / len(filled)
        width = len(filled)
        nxt = [len([c for c in rr if str(c).strip()]) for rr in rows[i + 1:i + 4]]
        if texty >= 0.8 and nxt and all(abs(w - width) <= max(1, width // 4) for w in nxt):
            return i
    return best


def _is_num(s: str) -> bool:
    try:
        float(s.replace(".", "").replace(",", ""))
        return True
    except ValueError:
        return False


def profile_columns(header: list, body: list[list]) -> list[dict]:
    out = []
    for j, name in enumerate(header):
        vals = [r[j] if j < len(r) else None for r in body]
        ci = infer_column(vals, str(name))
        non = [v for v in vals if v not in (None, "")]
        out.append({"name": str(name), "kind": ci.kind, "null_rate": round(1 - len(non) / max(1, len(vals)), 3),
                    "distinct": len(set(map(str, non))), "examples": list(dict.fromkeys(map(str, non)))[:5],
                    "number_format": ci.number_format.__dict__ if ci.number_format else None,
                    "units": ci.units, "notes": ci.notes})
    return out


def sniff_xlsx(path: str, max_rows: int = 3000) -> dict:
    from python_calamine import CalamineWorkbook
    wb = CalamineWorkbook.from_path(path)
    sheets = []
    for name in wb.sheet_names:
        rows = wb.get_sheet_by_name(name).to_python()
        hdr = guess_header(rows[:30]) if rows else 0
        sheets.append({"sheet": name, "n_rows": max(0, len(rows) - hdr - 1), "header_row": hdr,
                       "n_cols": max((len(r) for r in rows), default=0),
                       "preamble": [[str(c) for c in r if str(c).strip()] for r in rows[:hdr]],
                       "columns": profile_columns(rows[hdr] if rows else [], rows[hdr + 1:hdr + 1 + max_rows])})
    return {"sheets": sheets}


def sniff_sqlite(path: str, max_rows: int = 3000) -> dict:
    con = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)    # /data is read-only
    tables = []
    for (name, typ) in con.execute("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') "
                                   "AND name NOT LIKE 'sqlite_%' ORDER BY name"):
        cur = con.execute(f'SELECT * FROM "{name}" LIMIT {max_rows}')
        header = [d[0] for d in cur.description]
        body = [list(r) for r in cur.fetchall()]
        n = con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        fks = con.execute(f'PRAGMA foreign_key_list("{name}")').fetchall()
        tables.append({"table": name, "type": typ, "n_rows": n, "columns": profile_columns(header, body),
                       "foreign_keys": [{"from": f[3], "to": f"{f[2]}.{f[4]}"} for f in fks]})
    con.close()
    return {"tables": tables}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--out", default="reports/inventory")
    a = ap.parse_args()
    data_dir = os.path.join(a.split_dir, "data")
    inv = {"split": os.path.abspath(a.split_dir), "files_at_root": sorted(os.listdir(a.split_dir)), "businesses": {}}
    ext_count: Counter = Counter()
    for biz in sorted(os.listdir(data_dir)):
        bdir = os.path.join(data_dir, biz)
        if not os.path.isdir(bdir):
            continue
        entry = {"files": []}
        for fn in sorted(os.listdir(bdir)):
            fp = os.path.join(bdir, fn)
            ext = os.path.splitext(fn)[1].lower()
            ext_count[ext] += 1
            info = {"file": fn, "bytes": os.path.getsize(fp)}
            try:
                if ext in (".csv", ".tsv", ".txt"):
                    info.update(sniff_csv(fp))
                elif ext in (".xlsx", ".xlsm", ".xls"):
                    info.update(sniff_xlsx(fp))
                elif ext in (".sqlite", ".db", ".sqlite3"):
                    info.update(sniff_sqlite(fp))
                elif fn.lower() == "readme.md":
                    txt = open(fp, encoding="utf-8", errors="replace").read()
                    info.update({"chars": len(txt), "headings": [ln for ln in txt.splitlines() if ln.startswith("#")][:30]})
            except Exception as e:  # noqa: BLE001 - inventory must never crash
                info["error"] = f"{type(e).__name__}: {e}"
            entry["files"].append(info)
        inv["businesses"][biz] = entry
    inv["extensions"] = dict(ext_count)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(inv, f, ensure_ascii=False, indent=1, default=str)
    with open(os.path.join(a.out, "inventory.md"), "w", encoding="utf-8") as f:
        f.write(f"# Inventory {inv['split']}\n\nroot: {inv['files_at_root']}\n\nextensions: {dict(ext_count)}\n\n")
        for biz, e in inv["businesses"].items():
            f.write(f"## {biz}\n")
            for fi in e["files"]:
                f.write(f"- **{fi['file']}** ({fi['bytes']:,} B) {fi.get('error', '')}\n")
                for t in fi.get("tables", []) + fi.get("sheets", []) + ([fi] if "columns" in fi else []):
                    label = t.get("table") or t.get("sheet") or "csv"
                    cols = ", ".join(f"{c['name']}:{c['kind']}" for c in t["columns"])
                    f.write(f"  - `{label}` rows={t['n_rows']:,} hdr@{t.get('header_row', 0)} — {cols}\n")
    print(f"wrote {a.out}/inventory.md and inventory.json ({len(inv['businesses'])} businesses)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
