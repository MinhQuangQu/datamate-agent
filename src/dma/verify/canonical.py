"""Canonical form of a candidate (05.7 §1): two candidates are equal iff the official
scoring rules would score them equal; strings collapse to ENTITY identity (code)."""
from __future__ import annotations

from typing import Any

from dma.textnorm import scorer_norm


def entity_key(s: Any, vindex) -> str:
    e = vindex.lookup_exact(str(s)) if vindex is not None else None
    return f"@{e.kind}:{e.code or scorer_norm(e.name)}" if e else scorer_norm(s)


def elem_key(x: Any, vindex) -> str:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return f"{round(float(x), 2):.2f}"
    return entity_key(str(x), vindex)


def canonicalize(rec: dict, vindex=None) -> str:
    t = rec["type"]
    if t in ("unanswerable", "abstain"):
        return t
    v = rec.get("answer")
    if t in ("number", "percent"):
        return f"{t}:{round(float(v), 2):.2f}"
    if t == "string":
        return f"string:{entity_key(v, vindex)}"
    if t in ("date", "boolean"):
        return f"{t}:{str(v).lower()}"
    if t == "list":
        return "list:" + "|".join(elem_key(x, vindex) for x in v)
    if t == "table":
        rows = sorted("|".join(elem_key(c, vindex) for c in r) for r in v["rows"])
        return f"table:{len(v['columns'])}#" + ";".join(rows)
    return f"{t}:{v}"
