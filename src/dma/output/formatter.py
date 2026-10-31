"""Canonical answer -> submission record. The ONLY place that decides output format.

Rules (portal, Overview -> Evaluation / Data):
  number   int, or float rounded to 2 decimals          {"type":"number","answer":48250000}
  percent  percent UNITS (−12.45 means −12.45 %), 2 dp  {"type":"percent","answer":-12.45}
  string   name exactly as in the data (or object code)
  date     "yyyy-mm-dd", or "yyyy-mm" for month questions
  boolean  JSON true/false
  list     ordered; elements follow string/number rules
  table    {"columns":[...], "rows":[[...], ...]}
  unanswerable  optional "reason";  abstain -> {"id","type":"abstain"}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Any

from dma.textnorm import nfc

TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain")


class FormatError(ValueError):
    pass


@dataclass(frozen=True)
class FormatConfig:
    # How the reference generator rounded is NOT documented: verify on train answers
    # (EDA step "rounding-mode probe") and pin it here.
    rounding: str = "python"          # python | half_up | half_even
    max_list_len: int = 100
    max_table_rows: int = 200


def _to_float(x: Any) -> float:
    if isinstance(x, bool):
        raise FormatError("boolean where a number was expected")
    try:
        import numpy as np  # optional
        if isinstance(x, np.generic):
            x = x.item()
    except ImportError:
        pass
    if isinstance(x, Decimal):
        x = float(x)
    if not isinstance(x, (int, float)):
        raise FormatError(f"not a number: {x!r}")
    if isinstance(x, float) and not math.isfinite(x):
        raise FormatError("non-finite number")
    return float(x)


def round2(x: Any, mode: str = "python") -> float:
    v = _to_float(x)
    if mode == "python":
        r = round(v, 2)
    else:
        q = Decimal(repr(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP if mode == "half_up" else ROUND_HALF_EVEN)
        r = float(q)
    return 0.0 if r == 0 else r          # no "-0.0"


def fmt_number(x: Any, cfg: FormatConfig) -> int | float:
    r = round2(x, cfg.rounding)
    return int(r) if float(r).is_integer() and abs(r) < 2**53 else r


def fmt_date(x: Any, month_only: bool = False) -> str:
    if isinstance(x, datetime):
        x = x.date()
    if isinstance(x, date):
        return x.strftime("%Y-%m") if month_only else x.isoformat()
    if isinstance(x, str):
        s = x.strip()
        if len(s) == 7 and s[4] == "-":
            date(int(s[:4]), int(s[5:7]), 1)
            return s
        d = date.fromisoformat(s[:10])
        return d.strftime("%Y-%m") if month_only else d.isoformat()
    try:  # pandas Timestamp / numpy datetime64
        import pandas as pd
        return fmt_date(pd.Timestamp(x).to_pydatetime(), month_only)
    except Exception as e:  # noqa: BLE001
        raise FormatError(f"not a date: {x!r}") from e


def fmt_scalar_cell(x: Any, cfg: FormatConfig) -> Any:
    if x is None:
        raise FormatError("null cell")
    if isinstance(x, bool):
        return x
    try:
        return fmt_number(x, cfg)
    except FormatError:
        pass
    if isinstance(x, (date, datetime)):
        return fmt_date(x)
    return nfc(str(x)).strip()


def format_answer(qid: str, atype: str, value: Any = None, reason: str | None = None,
                  month_only: bool = False, cfg: FormatConfig = FormatConfig()) -> dict:
    if atype not in TYPES:
        raise FormatError(f"unknown type {atype}")
    rec: dict[str, Any] = {"id": qid, "type": atype}
    if atype == "abstain":
        return rec
    if atype == "unanswerable":
        if reason:
            rec["reason"] = str(reason)[:300]
        return rec
    if value is None:
        raise FormatError("missing value")
    if atype in ("number", "percent"):
        rec["answer"] = fmt_number(value, cfg)
    elif atype == "string":
        s = nfc(str(value)).strip()
        if not s:
            raise FormatError("empty string answer")
        rec["answer"] = s
    elif atype == "date":
        rec["answer"] = fmt_date(value, month_only)
    elif atype == "boolean":
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            value = value.strip().lower() == "true"
        try:
            import numpy as np
            if isinstance(value, np.bool_):
                value = bool(value)
        except ImportError:
            pass
        if not isinstance(value, bool):
            raise FormatError(f"not a boolean: {value!r}")
        rec["answer"] = value
    elif atype == "list":
        if not isinstance(value, (list, tuple)) or not value:
            raise FormatError("list answer must be a non-empty list")
        if len(value) > cfg.max_list_len:
            raise FormatError("list too long")
        rec["answer"] = [fmt_scalar_cell(v, cfg) for v in value]
    elif atype == "table":
        cols, rows = value.get("columns"), value.get("rows")
        if not cols or rows is None or len(rows) > cfg.max_table_rows:
            raise FormatError("bad table")
        if any(len(r) != len(cols) for r in rows):
            raise FormatError("ragged table")
        rec["answer"] = {"columns": [str(c) for c in cols],
                         "rows": [[fmt_scalar_cell(c, cfg) for c in r] for r in rows]}
    return rec


def safe_format(qid: str, atype: str, value: Any = None, **kw) -> tuple[dict, str | None]:
    """Never raises: falls back to abstain and returns the error message."""
    try:
        return format_answer(qid, atype, value, **kw), None
    except (FormatError, ValueError, TypeError, AttributeError) as e:
        return {"id": qid, "type": "abstain"}, f"{type(e).__name__}: {e}"
