"""Split discovery: find questions.jsonl, rulebook, calendar and business folders under /data.

Layout (portal, Data tab):  <root>/data/<business_id>/*  <root>/rulebook.md  <root>/calendar.csv
<root>/questions.jsonl. The root may sit up to 3 levels deep. (Gold answers are
looked up only by evaluation code: dma.eval.data.)
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import date

from dma.io.finalize import find_questions
from dma.types import Question


@dataclass
class SplitPaths:
    root: str
    questions_path: str
    rulebook_path: str | None
    calendar_path: str | None
    data_dir: str
    businesses: dict[str, str] = field(default_factory=dict)    # business_id -> dir
    warnings: list[str] = field(default_factory=list)


def _find_ci(root: str, name: str) -> str | None:
    for fn in sorted(os.listdir(root)):
        if fn.lower() == name.lower() and os.path.isfile(os.path.join(root, fn)):
            return os.path.join(root, fn)
    return None


def discover_split(data_root: str) -> SplitPaths:
    qpath = find_questions(data_root)
    root = os.path.dirname(qpath)
    warnings = []
    rb = _find_ci(root, "rulebook.md")
    cal = _find_ci(root, "calendar.csv")
    if rb is None:
        warnings.append("rulebook.md not found -> reference RuleSpec used")
    if cal is None:
        warnings.append("calendar.csv not found -> lunar algorithm used")
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(data_dir):
        data_dir = root
        warnings.append("no data/ folder: business folders searched next to questions.jsonl")
    businesses = {}
    for name in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, name)
        if os.path.isdir(p) and not name.startswith("."):
            businesses[name] = p
    return SplitPaths(root, qpath, rb, cal, data_dir, businesses, warnings)


def load_questions(path: str) -> tuple[list[Question], list[str]]:
    """UTF-8 (BOM ok), blank lines skipped, malformed lines reported (FR-IN-02)."""
    out, errors, seen = [], [], set()
    with open(path, encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                d = json.loads(line)
                q = Question(id=str(d["id"]), business_id=str(d["business_id"]), question=str(d["question"]),
                             as_of_date=date.fromisoformat(str(d["as_of_date"])[:10]))
            except Exception as e:  # noqa: BLE001
                errors.append(f"line {i}: {type(e).__name__}: {e}")
                continue
            if q.id in seen:
                errors.append(f"line {i}: duplicate id {q.id}")
                continue
            seen.add(q.id)
            out.append(q)
    return out, errors


def sha256_file(path: str | None) -> str | None:
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
