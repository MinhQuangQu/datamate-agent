"""Merge partial reruns into a base submission: later files win for every id they answered (abstain lines of a
partial run are ignored unless the id is absent from the base). Dev tool.

    python scripts/merge_submissions.py --base runs/v3/full/submission.jsonl --over runs/v3/p_009_010 --questions \
        data/raw/public_test/questions.jsonl --only-business dev_009,dev_010 --out submissions/public_v3.jsonl
"""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--over", nargs="*", default=[], help="submission.jsonl files of partial reruns")
    ap.add_argument("--questions", required=True)
    ap.add_argument("--only-business", default="", help="take over-file answers only for these shops")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8")]
    biz = {q["id"]: q["business_id"] for q in qs}
    keep = set(filter(None, a.only_business.split(",")))
    sub = {json.loads(l)["id"]: json.loads(l) for l in open(a.base, encoding="utf-8")}
    n = 0
    for f in a.over:
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if keep and biz.get(r["id"]) not in keep:
                continue
            sub[r["id"]] = r
            n += 1
    with open(a.out, "w", encoding="utf-8") as g:
        for q in qs:
            g.write(json.dumps(sub.get(q["id"], {"id": q["id"], "type": "abstain"}), ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(qs)} lines, {n} replaced from partial runs")


if __name__ == "__main__":
    main()
