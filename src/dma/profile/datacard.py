"""Data Card: the LLM-facing rendering of profile.json (05.1 §10). Budget ~2.5k tokens:
examples are cut first, columns never."""
from __future__ import annotations

from typing import Any


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}".rstrip("0").rstrip(".") if abs(v) < 1e15 else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def approx_tokens(s: str) -> int:
    return max(1, len(s) // 3)


def render_data_card(prof: dict, readme: str | None, max_tokens: int = 2500, examples: int = 3) -> str:
    for ex in (examples, 2, 1, 0):
        card = _render(prof, readme, ex)
        if approx_tokens(card) <= max_tokens:
            return card
    return card


def _render(prof: dict, readme: str | None, n_ex: int) -> str:
    out = ["# DATA CARD"]
    files = ", ".join(f"{f['file']}" + (f" (tables: {', '.join(f['tables'])})" if f.get("tables") else "")
                      for f in prof.get("files", []))
    out.append(f"files: {files}")
    if readme is not None:
        out.append(f"README: {len(readme)} chars")
    if prof.get("date_range"):
        out.append(f"data range (all date columns): {prof['date_range'][0]} .. {prof['date_range'][1]}")
    fk_by = {}
    for fk in prof.get("fks", []):
        fk_by.setdefault((fk["from_table"], fk["from_column"]), []).append(fk)
    for t in prof["tables"]:
        hdr = f"## typed.{t['name']}  ({t['n_rows']:,} rows; grain≈{t.get('grain', '?')}; from {t['source']}"
        if t.get("primary_key"):
            hdr += f"; key {','.join(t['primary_key'])}"
        out.append(hdr + ")")
        for c in t.get("columns", []):
            if c["kind"] == "empty":
                continue
            s = f"- {c['name']} \"{c['orig_name']}\" {c['kind']}"
            if c.get("distinct") is not None:
                s += f" distinct {c['distinct']:,}"
            if c["sql_type"] in ("BIGINT", "DOUBLE", "DATE", "TIMESTAMP") and c.get("min") is not None:
                s += f" {_fmt(c['min'])}..{_fmt(c['max'])}"
            if c.get("has_time"):
                s += " (has time of day)"
            if c.get("number_format"):
                s += f" {c['number_format']} format"
            if c.get("units"):
                s += f" (units: {', '.join(c['units'][:4])})"
            if c.get("unit_multiplier", 1.0) != 1.0:
                s += f" ×{c['unit_multiplier']:g} (unit)"
            top = c.get("top")
            if top and c["kind"] in ("string", "bool") and c.get("distinct", 99) <= 12:
                tot = sum(k for _, k in top) or 1
                s += " {" + ", ".join(f"{_fmt(v)}: {100 * k / tot:.1f}%" for v, k in top[:8]) + "}"
            elif n_ex and c.get("examples"):
                s += " e.g. " + ", ".join(_fmt(e) for e in c["examples"][:n_ex])
            if c.get("null_rate", 0) > 0.05:
                s += f" null {100 * c['null_rate']:.0f}%"
            if c.get("parse_fail"):
                s += f" ⚠ {c['parse_fail']} unparsed cells"
            for fk in fk_by.get((t["name"], c["name"]), [])[:1]:
                s += f"  FK? -> typed.{fk['to_table']}.{fk['to_column']} ({100 * fk['coverage']:.1f}%)"
            out.append(s)
        for idn in t.get("identities", [])[:3]:
            out.append(f"  identity: {idn['kind']} on {', '.join(idn['cols'])} ({100 * idn['share']:.1f}% rows)")
        notes = []
        if t.get("preamble"):
            notes.append("preamble " + " | ".join(f'"{p[:60]}"' for p in t["preamble"][:3]))
        if t.get("summary_rows"):
            notes.append(f"{t['summary_rows']} summary rows removed")
        if t.get("duplicate_rows"):
            notes.append(f"{t['duplicate_rows']} exact duplicate rows kept")
        notes += t.get("warnings", [])[:3]
        if notes:
            out.append("notes: " + "; ".join(notes))
    return "\n".join(out)
