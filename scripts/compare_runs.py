"""Reproducibility check (Rule 8): compare 2+ runs of the same image.

    python scripts/compare_runs.py runs/repro_1 runs/repro_2 runs/repro_3 [--gold answers.jsonl]
Prints the share of questions whose canonical answer changed and, with --gold, PA of each run and ΔPA.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.eval.scorer import read_jsonl, score  # noqa: E402
from dma.textnorm import scorer_norm  # noqa: E402


def canon(rec: dict) -> str:
    t, a = rec.get("type"), rec.get("answer")
    if t in ("number", "percent"):
        return f"{t}:{round(float(a), 2):.2f}"
    if t == "string":
        return f"string:{scorer_norm(a)}"
    if t == "list":
        return "list:" + "|".join(f"{round(x, 2):.2f}" if isinstance(x, (int, float)) and not isinstance(x, bool)
                                  else scorer_norm(x) for x in a)
    if t == "table":
        rows = sorted("|".join(scorer_norm(c) for c in r) for r in a["rows"])
        return f"table:{len(a['columns'])}#" + ";".join(rows)
    return f"{t}:{json.dumps(a, ensure_ascii=False, sort_keys=True)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--gold")
    a = ap.parse_args()
    subs = {r: {x["id"]: x for x in read_jsonl(os.path.join(r, "submission.jsonl"))} for r in a.runs}
    ids = sorted(set().union(*[set(s) for s in subs.values()]))
    for r1, r2 in itertools.combinations(a.runs, 2):
        diff = [i for i in ids if canon(subs[r1].get(i, {"type": "abstain"})) != canon(subs[r2].get(i, {"type": "abstain"}))]
        print(f"{os.path.basename(r1)} vs {os.path.basename(r2)}: changed {len(diff)}/{len(ids)} = {len(diff) / max(1, len(ids)):.2%}")
        for i in diff[:10]:
            print("   ", i, canon(subs[r1].get(i, {"type": "abstain"})), "<->", canon(subs[r2].get(i, {"type": "abstain"})))
    if a.gold:
        gold = read_jsonl(a.gold)
        pas = {r: score(list(s.values()), gold).pa for r, s in subs.items()}
        for r, p in pas.items():
            print(f"PA {os.path.basename(r)} = {p:.4f}")
        print(f"max ΔPA = {max(pas.values()) - min(pas.values()):.4f}  (BTC tolerance 0.03, internal target 0.01)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
