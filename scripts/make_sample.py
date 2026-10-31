"""Build a stratified sample split (N questions per shop) from a split with answers, for quick scored runs.

    python scripts/make_sample.py --split data/raw/train --per-business 10 --out data/samples/train_s300

Shop folders are symlinked, not copied. Selection is a deterministic hash of the question text, so the
sample is stable across runs. Dev tool only: nothing in src/ reads it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--per-business", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    qs = [json.loads(l) for l in open(os.path.join(a.split, "questions.jsonl"), encoding="utf-8")]
    by_biz: dict[str, list[dict]] = defaultdict(list)
    for q in qs:
        by_biz[q["business_id"]].append(q)
    keep: set[str] = set()
    for items in by_biz.values():
        items.sort(key=lambda q: hashlib.sha256(q["question"].encode()).hexdigest())
        keep.update(q["id"] for q in items[: a.per_business])

    os.makedirs(os.path.join(a.out, "data"), exist_ok=True)
    for name in ("rulebook.md", "calendar.csv"):
        shutil.copy(os.path.join(a.split, name), os.path.join(a.out, name))
    for biz in by_biz:
        link = os.path.join(a.out, "data", biz)
        if not os.path.exists(link):
            os.symlink(os.path.abspath(os.path.join(a.split, "data", biz)), link)
    for name in ("questions.jsonl", "answers.jsonl"):
        src = os.path.join(a.split, name)
        if not os.path.exists(src):
            continue
        with open(src, encoding="utf-8") as f, open(os.path.join(a.out, name), "w", encoding="utf-8") as g:
            for line in f:
                if json.loads(line)["id"] in keep:
                    g.write(line)
    print(f"{a.out}: {len(keep)} questions, {len(by_biz)} shops")


if __name__ == "__main__":
    main()
