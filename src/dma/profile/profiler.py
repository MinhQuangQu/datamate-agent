"""Column/table profiling over typed.* (05.1 §9): stats, keys, FK candidates (inclusion
>= 95%), arithmetic identities (q*p≈a, g-d≈n, q*p-d≈a, q*p*(1-d%)≈a) and grain guess.
Output is a JSON-able dict (profile.json) — the evidence base for the BSM.
"""
from __future__ import annotations

import re
from itertools import permutations
from typing import Any

import numpy as np

from dma.ingest.loader import IngestResult, q
from dma.textnorm import fold

ID_RE = re.compile(r"^[A-Za-z]{1,6}[-_]?\d{1,8}$")
MAX_DISTINCT_FK = 200_000
SAMPLE_ROWS = 5000


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if v != v else float(v)
    return v


def profile_business(ing: IngestResult, con) -> dict:
    tables = []
    for lt in ing.tables:
        try:
            tables.append(_profile_table(con, lt))
        except Exception as e:  # noqa: BLE001
            tables.append({"name": lt.name, "source": lt.source, "n_rows": lt.n_rows, "columns": [],
                           "error": f"{type(e).__name__}: {e}"})
    fks = _fk_candidates(con, tables, ing)
    dates = [t["date_range"] for t in tables if t.get("date_range")]
    prof = {
        "tables": tables,
        "fks": fks,
        "date_range": [min(d[0] for d in dates), max(d[1] for d in dates)] if dates else None,
        "files": ing.files,
        "warnings": list(ing.warnings),
    }
    for t in tables:
        t["grain"] = _grain(t, fks)
    return prof


def _profile_table(con, lt) -> dict:
    tn = f"typed.{q(lt.name)}"
    n = lt.n_rows
    cols = []
    for cm in lt.columns:
        c = q(cm.name)
        d: dict[str, Any] = {"name": cm.name, "orig_name": cm.orig_name, "kind": cm.kind, "sql_type": cm.sql_type,
                             "null_rate": cm.null_rate, "unit_multiplier": cm.unit_multiplier, "units": cm.units,
                             "number_format": cm.number_format, "date_order": cm.date_order,
                             "parse_fail": cm.parse_fail}
        if cm.kind == "empty" or n == 0:
            d.update(distinct=0, distinct_ratio=0.0)
            cols.append(d)
            continue
        nn, distinct = con.execute(f"SELECT COUNT({c}), COUNT(DISTINCT {c}) FROM {tn}").fetchone()
        d["distinct"] = int(distinct)
        d["distinct_ratio"] = round(distinct / max(1, nn), 4)
        d["unique"] = bool(nn == n and distinct == nn and n > 0)
        if cm.sql_type in ("BIGINT", "DOUBLE", "DATE", "TIMESTAMP"):
            agg = f", AVG({c}), MEDIAN({c})" if cm.sql_type in ("BIGINT", "DOUBLE") else ""
            row = con.execute(f"SELECT MIN({c}), MAX({c}){agg} FROM {tn}").fetchone()
            d["min"], d["max"] = _jsonable(row[0]), _jsonable(row[1])
            if agg:
                d["mean"], d["median"] = _jsonable(row[2]), _jsonable(row[3])
            if cm.sql_type in ("DATE", "TIMESTAMP"):
                d["has_time"] = cm.sql_type == "TIMESTAMP" and bool(con.execute(
                    f"SELECT COUNT(*) FROM {tn} WHERE {c} IS NOT NULL AND (hour({c}) <> 0 OR minute({c}) <> 0)").fetchone()[0])
        if cm.sql_type in ("VARCHAR", "BOOLEAN") or (cm.sql_type == "BIGINT" and distinct <= 20):
            top = con.execute(f"SELECT {c}, COUNT(*) AS k FROM {tn} WHERE {c} IS NOT NULL GROUP BY 1 "
                              f"ORDER BY k DESC, 1 ASC LIMIT 12").fetchall()
            d["top"] = [[_jsonable(v), int(k)] for v, k in top]
        ex = con.execute(f"SELECT DISTINCT {c} FROM (SELECT {c} FROM {tn} WHERE {c} IS NOT NULL LIMIT 200) "
                         f"ORDER BY 1 LIMIT 5").fetchall()
        d["examples"] = [_jsonable(r[0]) for r in ex]
        if cm.sql_type == "VARCHAR":
            sample = [str(v) for v, _ in d.get("top", [])] + [str(e) for e in d["examples"]]
            d["id_like"] = bool(sample) and sum(bool(ID_RE.match(s)) for s in sample) >= 0.9 * len(sample)
            d["avg_len"] = _jsonable(con.execute(f"SELECT AVG(length({c})) FROM {tn}").fetchone()[0])
        cols.append(d)
    t = {"name": lt.name, "source": lt.source, "n_rows": n, "columns": cols, "header_row": lt.header_row,
         "preamble": lt.preamble, "notes": lt.notes, "warnings": lt.warnings, "is_view": lt.is_view,
         "summary_rows": lt.summary_rows, "duplicate_rows": lt.duplicate_rows, "declared_fks": lt.declared_fks}
    dcols = [c for c in cols if c["sql_type"] in ("DATE", "TIMESTAMP") and c.get("min")]
    if dcols:
        t["date_range"] = [min(c["min"][:10] for c in dcols), max(c["max"][:10] for c in dcols)]
    def key_rank(c: dict) -> int | None:
        if not c.get("unique") or c["sql_type"] not in ("VARCHAR", "BIGINT"):
            return None
        hint = bool(re.search(r"\b(ma|id|code|so|stt|sku)\b", fold(c["orig_name"])))
        if c["kind"] == "id" or (c["kind"] == "string" and c.get("id_like")):
            return 0 if hint else 1
        if hint and c["kind"] in ("number", "string"):
            return 2
        return None
    ranked = sorted((r, i, c["name"]) for i, c in enumerate(cols) if (r := key_rank(c)) is not None)
    t["primary_key"] = [ranked[0][2]] if ranked else []
    t["identities"] = _identities(con, tn, cols) if n else []
    return t


def _close(x: np.ndarray, y: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(x - y) <= np.maximum(tol, 1e-6 * np.abs(y))


def _identities(con, tn: str, cols: list[dict], tol: float = 1.0) -> list[dict]:
    num = [c["name"] for c in cols if c["sql_type"] in ("BIGINT", "DOUBLE") and c.get("distinct", 0) >= 2][:14]
    if len(num) < 2:
        return []
    sel = ", ".join(f"CAST({q(c)} AS DOUBLE)" for c in num)
    data = np.array(con.execute(f"SELECT {sel} FROM {tn} LIMIT {SAMPLE_ROWS}").fetchall(), dtype=float)
    if data.size == 0:
        return []
    A = {c: data[:, i] for i, c in enumerate(num)}
    out = []

    def share(mask_ok, valid):
        k = valid.sum()
        return float((mask_ok & valid).sum() / k) if k >= 3 else 0.0

    found_mult = set()
    for a in num:
        ya = A[a]
        for qc, pc in permutations([c for c in num if c != a], 2):
            if qc > pc:
                continue
            x = A[qc] * A[pc]
            valid = ~np.isnan(x) & ~np.isnan(ya) & (ya != 0)
            s = share(_close(x, ya, tol), valid)
            if s >= 0.95:
                out.append({"kind": "q*p=a", "cols": [qc, pc, a], "share": round(s, 4)})
                found_mult.add(a)
    for nc in num:
        for g, dcol in permutations([c for c in num if c != nc], 2):
            x = A[g] - A[dcol]
            valid = ~np.isnan(x) & ~np.isnan(A[nc]) & (A[nc] != 0) & (A[dcol] != 0)
            if valid.sum() < 3:
                continue
            s = share(_close(x, A[nc], tol), valid)
            if s >= 0.95:
                out.append({"kind": "g-d=n", "cols": [g, dcol, nc], "share": round(s, 4)})
    if len(num) <= 10:
        for a in num:
            if a in found_mult:
                continue
            others = [c for c in num if c != a]
            for qc, pc in permutations(others, 2):
                if qc > pc:
                    continue
                for dcol in others:
                    if dcol in (qc, pc):
                        continue
                    base = A[qc] * A[pc]
                    nz = ~np.isnan(A[dcol]) & (A[dcol] != 0)
                    valid = ~np.isnan(base) & ~np.isnan(A[a]) & (A[a] != 0) & ~np.isnan(A[dcol])
                    if nz.sum() < 3:
                        continue
                    s1 = share(_close(base - A[dcol], A[a], tol), valid)
                    if s1 >= 0.95:
                        out.append({"kind": "q*p-d=a", "cols": [qc, pc, dcol, a], "share": round(s1, 4)})
                        continue
                    if np.nanmax(np.abs(A[dcol])) <= 100:
                        s2 = share(_close(base * (1 - A[dcol] / 100.0), A[a], tol), valid)
                        if s2 >= 0.95:
                            out.append({"kind": "q*p*(1-d%)=a", "cols": [qc, pc, dcol, a], "share": round(s2, 4)})
    return out[:20]


def _norm_key(v: Any) -> str:
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v).strip().upper()


def _fk_candidates(con, tables: list[dict], ing: IngestResult) -> list[dict]:
    declared = []
    for lt in ing.tables:
        for fc, tt, tc in lt.declared_fks:
            declared.append({"from_table": lt.name, "from_column": fc, "to_table": tt, "to_column": tc,
                             "coverage": 1.0, "declared": True})
    keys = []   # (table, col, valueset)
    cands = []  # (table, col, valueset)
    cache: dict[tuple[str, str], set[str]] = {}

    def values(t: str, c: str) -> set[str]:
        k = (t, c)
        if k not in cache:
            rows = con.execute(f"SELECT DISTINCT {q(c)} FROM typed.{q(t)} WHERE {q(c)} IS NOT NULL").fetchall()
            cache[k] = {_norm_key(r[0]) for r in rows}
        return cache[k]

    for t in tables:
        for c in t["columns"]:
            if c["kind"] not in ("id", "string", "number") or c["sql_type"] not in ("VARCHAR", "BIGINT"):
                continue
            if not c.get("distinct") or c["distinct"] > MAX_DISTINCT_FK:
                continue
            if c["kind"] == "string" and (c.get("avg_len") or 0) > 60:
                continue
            if c.get("unique") and t["n_rows"] >= 2:
                keys.append((t["name"], c))
            if c["distinct"] >= 1:
                cands.append((t["name"], c))
    out = []
    for ft, fc in cands:
        best = []
        for kt, kc in keys:
            if kt == ft:
                continue
            if fc["kind"] != kc["kind"] and not ({fc["kind"], kc["kind"]} <= {"id", "string"}):
                continue
            if kc["distinct"] < fc["distinct"]:
                continue
            fv = values(ft, fc["name"])
            if not fv:
                continue
            kv = values(kt, kc["name"])
            cov = len(fv & kv) / len(fv)
            if cov >= 0.95:
                name_sim = _name_sim(fc, kc)
                if fc["kind"] == "number" and name_sim < 0.5:
                    continue          # integer columns overlap by accident (quantities vs ids)
                best.append({"from_table": ft, "from_column": fc["name"], "to_table": kt, "to_column": kc["name"],
                             "coverage": round(cov, 4), "name_sim": round(name_sim, 3), "declared": False})
        best.sort(key=lambda r: (-(r["coverage"] + 0.5 * r["name_sim"]), r["to_table"], r["to_column"]))
        out += best[:2]
    seen = {(d["from_table"], d["from_column"], d["to_table"]) for d in declared}
    return declared + [r for r in out if (r["from_table"], r["from_column"], r["to_table"]) not in seen]


def _name_sim(a: dict, b: dict) -> float:
    from rapidfuzz import fuzz
    x, y = fold(a["orig_name"]), fold(b["orig_name"])
    return max(fuzz.token_set_ratio(x, y), fuzz.ratio(a["name"], b["name"])) / 100.0


def _grain(t: dict, fks: list[dict]) -> str:
    cols = t.get("columns", [])
    ids = [c for c in cols if c["kind"] == "id" or (c["kind"] == "string" and c.get("id_like"))]
    has_date = any(c["sql_type"] in ("DATE", "TIMESTAMP") for c in cols)
    outgoing = [f for f in fks if f["from_table"] == t["name"]]
    if t.get("primary_key"):
        return "event" if has_date and outgoing else ("master" if not has_date else "event_or_master")
    repeated = [c for c in ids if not c.get("unique")]
    if len(repeated) >= 2 or (repeated and outgoing):
        return "line"
    return "other"
