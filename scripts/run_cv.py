"""Leave-business-out CV (08 §2): run dma.run per fold (exemplars of the fold's businesses
excluded), merge, score, break down, export out-of-fold features for the calibrator.

    python scripts/run_cv.py --split data/raw/train --folds 5 --config configs/dev.yaml \
        --endpoints http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1 --out runs/$(date +%m%d_%H%M)_lobo
    python scripts/run_cv.py --split data/synth/v1 --folds 1 --fake-llm --out runs/synth_fake
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from dma.eval.scorer import read_jsonl, score  # noqa: E402
from dma.eval.data import answers_path  # noqa: E402
from dma.io.split import discover_split, load_questions  # noqa: E402


def folds_of(businesses: list[str], k: int) -> list[list[str]]:
    bs = sorted(businesses)
    if k <= 1:
        return [bs]
    return [bs[i::k] for i in range(k)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "dev.yaml"))
    ap.add_argument("--profile")
    ap.add_argument("--endpoints")
    ap.add_argument("--fake-llm", action="store_true")
    ap.add_argument("--limit-per-fold", type=int)
    ap.add_argument("--compare", help="previous run dir: writes diff.csv")
    ap.add_argument("--only-fold", type=int)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    sp = discover_split(a.split)
    questions, _ = load_questions(sp.questions_path)
    biz = sorted({q.business_id for q in questions})
    folds = folds_of(biz, a.folds)
    env = dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep + ROOT)
    for i, fb in enumerate(folds):
        if a.only_fold is not None and i != a.only_fold:
            continue
        rd = os.path.join(a.out, f"fold_{i}")
        cmd = [sys.executable, "-m", "dma.run", "--config", a.config, "--data", a.split, "--result", rd,
               "--work", os.path.join(a.out, "work", f"fold_{i}"), "--only-business", ",".join(fb),
               "--exclude-memory-businesses", ",".join(fb)]
        if a.profile:
            cmd += ["--profile", a.profile]
        if a.endpoints:
            cmd += ["--endpoints", a.endpoints]
        if a.fake_llm:
            cmd += ["--fake-llm"]
        if a.limit_per_fold:
            cmd += ["--limit", str(a.limit_per_fold)]
        print(f"[cv] fold {i}: {len(fb)} businesses", flush=True)
        subprocess.run(cmd, env=env, check=False)
    # merge (each fold's submission also carries abstain lines for other folds' ids -> keep fold-owned ones)
    qbiz = {q.id: q.business_id for q in questions}
    preds: dict[str, dict] = {}
    oof = []
    for i, fb in enumerate(folds):
        rd = os.path.join(a.out, f"fold_{i}")
        sub = os.path.join(rd, "submission.jsonl")
        if not os.path.exists(sub):
            continue
        owned = set(fb)
        for r in read_jsonl(sub):
            if qbiz.get(r["id"]) in owned and (r["id"] not in preds or preds[r["id"]]["type"] == "abstain"):
                preds[r["id"]] = r
    golds = read_jsonl(answers_path(a.split)) if answers_path(a.split) else []
    with open(os.path.join(a.out, "submission.jsonl"), "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(preds.get(q.id, {"id": q.id, "type": "abstain"}), ensure_ascii=False) + "\n")
    if not golds:
        print("[cv] no answers.jsonl: merged submission written, no score")
        return 0
    rep = score(list(preds.values()), golds)
    gold = {g["id"]: g for g in golds}
    by = defaultdict(lambda: [0, 0, 0])
    rows = []
    for it in rep.items:
        tr = _trace(a.out, folds, qbiz, it.id)
        src = (tr.get("decision") or {}).get("source", "none")
        keys = {"type": gold[it.id]["type"], "business": qbiz.get(it.id, "?"), "source": src,
                "chosen": (tr.get("decision") or {}).get("chosen_solver") or "-"}
        for k, v in keys.items():
            cell = by[(k, v)]
            cell[0] += 1
            cell[1] += it.score
            cell[2] += int(it.score == -1)
        rows.append({"id": it.id, "score": it.score, "reason": it.reason, **keys,
                     "question": tr.get("question", ""), "pred": json.dumps(preds.get(it.id), ensure_ascii=False),
                     "gold": json.dumps(gold[it.id].get("answer"), ensure_ascii=False)})
        if tr.get("features") and (tr.get("decision") or {}).get("chosen_solver"):
            chosen_correct = _chosen_correct(tr, gold[it.id])
            if chosen_correct is not None:
                oof.append({"id": it.id, "business_id": qbiz.get(it.id), "features": tr["features"], "y": chosen_correct,
                            "answer_type": gold[it.id]["type"], "source": src})
    with open(os.path.join(a.out, "items.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "oof_features.jsonl"), "w", encoding="utf-8") as f:
        for r in oof:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lines = [f"# CV summary — {a.split}", "", f"N={rep.n}  correct={rep.correct}  wrong={rep.wrong}  abstain={rep.abstain}  "
             f"**PA={rep.pa:.4f}**", ""]
    for dim in ("type", "source", "chosen", "business"):
        lines += [f"## by {dim}", "", "| value | n | PA | wrong |", "|---|---:|---:|---:|"]
        for (k, v), (n, s, wr) in sorted(by.items()):
            if k == dim:
                lines.append(f"| {v} | {n} | {s / n:+.3f} | {wr} |")
        lines.append("")
    if a.compare and os.path.exists(os.path.join(a.compare, "items.csv")):
        prev = {r["id"]: r for r in csv.DictReader(open(os.path.join(a.compare, "items.csv"), encoding="utf-8"))}
        diff = [r for r in rows if r["id"] in prev and int(prev[r["id"]]["score"]) != r["score"]]
        with open(os.path.join(a.out, "diff.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id", "prev", "now", "question", "pred", "gold"])
            w.writeheader()
            for r in diff:
                w.writerow({"id": r["id"], "prev": prev[r["id"]]["score"], "now": r["score"], "question": r["question"],
                            "pred": r["pred"], "gold": r["gold"]})
        lines.append(f"diff vs {a.compare}: {len(diff)} changed items "
                     f"(+{sum(1 for r in diff if r['score'] > int(prev[r['id']]['score']))} better)")
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:4]))
    return 0


def _trace(out: str, folds, qbiz, qid: str) -> dict:
    for i, fb in enumerate(folds):
        if qbiz.get(qid) in fb:
            p = os.path.join(out, f"fold_{i}", "logs", f"{''.join(c if c.isalnum() or c in '-_.' else '_' for c in qid)}.json")
            if os.path.exists(p):
                try:
                    return json.load(open(p, encoding="utf-8"))
                except ValueError:
                    return {}
    return {}


def _chosen_correct(tr: dict, gold: dict) -> int | None:
    """Correctness of the candidate the system WOULD answer (even if it abstained) -> calibrator label."""
    from dma.eval.scorer import score_item
    chosen = (tr.get("decision") or {}).get("chosen_solver")
    cand = next((c for c in tr.get("candidates", []) if c.get("solver") == chosen), None)
    if cand is None:
        return None
    rec = {"id": gold["id"], "type": cand["answer_type"]}
    if cand["answer_type"] != "unanswerable":
        rec["answer"] = cand["value"]
    s, _ = score_item(rec, gold)
    return int(s == 1)


if __name__ == "__main__":
    raise SystemExit(main())
