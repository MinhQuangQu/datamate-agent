"""Train the PA-optimal abstention calibrator on out-of-fold features (05.7 §6.2).

    python scripts/train_calibrator.py runs/1105_lobo/oof_features.jsonl [more.jsonl ...] --out assets/calibrator.json

Label = the candidate the system WOULD answer is correct (from run_cv.py). Thresholds:
tau = middle of the PA plateau (>= 0.5) on answers; tau_u the same on `unanswerable` answers.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.verify.calibrator import Calibrator, choose_threshold, expected_calibration_error, pa_curve  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("oof", nargs="+")
    ap.add_argument("--out", default="assets/calibrator.json")
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--min-tau-u", type=float, default=0.6)
    ap.add_argument("--extra-margin", type=float, default=0.0, help="raise tau for distribution shift (05.7 §6.2.5)")
    a = ap.parse_args(argv)
    X, y, unans = [], [], []
    for p in a.oof:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    X.append(r["features"])
                    y.append(int(r["y"]))
                    unans.append(bool(r["features"].get("unanswerable_pred")))
    if len(set(y)) < 2 or len(y) < 30:
        print(f"[calibrator] not enough data ({len(y)} rows, classes {set(y)}) — keeping fallback rules")
        return 1
    cal = Calibrator().fit(X, y, C=a.C)
    probs = [cal.prob(x) for x in X]
    pa_idx = [i for i, u in enumerate(unans) if not u]
    u_idx = [i for i, u in enumerate(unans) if u]
    tau, best = choose_threshold([probs[i] for i in pa_idx], [y[i] for i in pa_idx])
    tau_u = a.min_tau_u
    if len(u_idx) >= 10:
        tu, _ = choose_threshold([probs[i] for i in u_idx], [y[i] for i in u_idx])
        tau_u = max(tu, a.min_tau_u)
    cal.threshold = round(min(0.95, tau + a.extra_margin), 4)
    cal.threshold_unanswerable = round(min(0.97, tau_u + a.extra_margin), 4)
    cal.save(a.out)
    answer_all = pa_curve(probs, y, __import__("numpy").array([0.0]))[0][1]
    print(json.dumps({"n": len(y), "base_acc": round(sum(y) / len(y), 4), "ece": round(expected_calibration_error(probs, y), 4),
                      "tau": cal.threshold, "tau_u": cal.threshold_unanswerable, "pa_answer_all": round(answer_all, 4),
                      "pa_at_tau": round(best, 4), "out": a.out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
