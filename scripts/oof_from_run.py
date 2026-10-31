"""Calibrator rows from an ordinary scored run (no CV folds), same format as run_cv.py's oof_features.jsonl.

    python scripts/oof_from_run.py --run runs/sub300 --split data/samples/train_s300 --out runs/sub300/oof_features.jsonl

Valid as out-of-fold only while the exemplar memory holds no question of the run's businesses
(assets/exemplars.jsonl = hand-written seeds today). Re-check that before using mined exemplars.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(__file__))
from dma.eval.data import answers_path  # noqa: E402
from dma.eval.scorer import read_jsonl  # noqa: E402
from run_cv import _chosen_correct  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gold = {g["id"]: g for g in read_jsonl(answers_path(a.split))}
    rows = []
    for p in sorted(glob.glob(os.path.join(a.run, "logs", "*.json"))):
        try:
            tr = json.load(open(p, encoding="utf-8"))
        except ValueError:
            continue
        if not isinstance(tr, dict) or tr.get("id") not in gold:
            continue
        if not (tr.get("features") and (tr.get("decision") or {}).get("chosen_solver")):
            continue
        y = _chosen_correct(tr, gold[tr["id"]])
        if y is not None:
            rows.append({"id": tr["id"], "business_id": tr.get("business_id"), "features": tr["features"], "y": y,
                         "answer_type": gold[tr["id"]]["type"], "source": tr["decision"].get("source")})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[oof] {len(rows)} rows, {sum(r['y'] for r in rows)} correct -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
