"""Rule 4 guard: fail if source code branches on business/question ids or reads answers at inference.

    python scripts/check_no_id_logic.py src/
"""
from __future__ import annotations

import os
import re
import sys

PATTERNS = [
    (r"\bbiz_\d+", "literal business id"),
    (r"\bq_[0-9a-f]{6,}", "literal question id"),
    (r"business_id\s*(==|!=|\bin\b)\s*['\"\[(]", "branch on business_id"),
    (r"\bid\s*(==|!=)\s*['\"]q_", "branch on question id"),
    (r"answers\.jsonl", "reads answers.jsonl (allowed only in eval/ and scripts/)"),
]
ALLOWED_DIRS = ("eval", "tests")


def scan(root: str) -> list[str]:
    hits = []
    for cur, _, files in os.walk(root):
        if any(part in ALLOWED_DIRS for part in cur.split(os.sep)):
            continue
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            path = os.path.join(cur, fn)
            for n, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
                if line.lstrip().startswith("#"):
                    continue
                for pat, why in PATTERNS:
                    if re.search(pat, line):
                        hits.append(f"{path}:{n}: {why}: {line.strip()[:120]}")
    return hits


if __name__ == "__main__":
    found = scan(sys.argv[1] if len(sys.argv) > 1 else "src")
    print("\n".join(found) if found else "OK: no id-driven logic found")
    sys.exit(1 if found else 0)
