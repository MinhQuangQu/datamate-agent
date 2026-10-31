"""Validate a submission against questions.jsonl (schema, completeness, duplicates, size).

    python scripts/validate_submission.py result/submission.jsonl data/questions.jsonl
Exit code 0 = valid and complete.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.io.finalize import read_ids, validate_record  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024


def main(sub_path: str, questions_path: str) -> int:
    ids = read_ids(questions_path)
    problems: list[str] = []
    seen: Counter = Counter()
    types: Counter = Counter()
    with open(sub_path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"line {n}: invalid JSON ({e})")
                continue
            why = validate_record(rec)
            if why:
                problems.append(f"line {n}: {why}")
            seen[rec.get("id")] += 1
            types[rec.get("type")] += 1
    missing = [i for i in ids if i not in seen]
    dups = [i for i, c in seen.items() if c > 1]
    unknown = [i for i in seen if i not in set(ids)]
    size = os.path.getsize(sub_path)
    print(f"questions={len(ids)} lines={sum(seen.values())} missing={len(missing)} duplicates={len(dups)} "
          f"unknown={len(unknown)} invalid={len(problems)} size={size / 1024:.1f}KB")
    print("types:", dict(types))
    for p in problems[:20]:
        print("  ", p)
    if size > MAX_BYTES:
        print("ERROR: file larger than 5 MB")
    ok = not (missing or dups or unknown or problems or size > MAX_BYTES)
    print("VALID" if ok else "INVALID")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
