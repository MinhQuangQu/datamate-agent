"""Evaluation-only data access (never imported by the inference path)."""
from __future__ import annotations

import os

from dma.io.split import discover_split


def answers_path(split_root: str) -> str | None:
    root = os.path.dirname(discover_split(split_root).questions_path)
    for fn in sorted(os.listdir(root)):
        if fn.lower() == "answers.jsonl":
            return os.path.join(root, fn)
    return None
