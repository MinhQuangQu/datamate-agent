"""Sanity statistics of a finished run (runbook 14 §4): answer types, abstain rate, timing, warnings.

    python scripts/run_stats.py /result
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import Counter


def main(result_dir: str) -> int:
    sub = os.path.join(result_dir, "submission.jsonl")
    recs = [json.loads(x) for x in open(sub, encoding="utf-8") if x.strip()]
    types = Counter(r["type"] for r in recs)
    n = len(recs)
    print(f"records={n}  abstain={types.get('abstain', 0)} ({types.get('abstain', 0) / max(1, n):.1%})  "
          f"unanswerable={types.get('unanswerable', 0)} ({types.get('unanswerable', 0) / max(1, n):.1%})")
    print("types:", dict(types.most_common()))
    times, sources, errors = [], Counter(), 0
    for p in glob.glob(os.path.join(result_dir, "logs", "*.json")):
        if os.path.basename(p) in ("run_manifest.json",):
            continue
        try:
            log = json.load(open(p, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if "timing_ms" in log:
            times.append(log["timing_ms"].get("total", 0) / 1000)
        sources[(log.get("decision") or {}).get("source", "?")] += 1
        errors += bool(log.get("errors"))
    if times:
        q = statistics.quantiles(times, n=20) if len(times) >= 20 else [max(times)] * 19
        print(f"per-question seconds: median={statistics.median(times):.1f} p95={q[18]:.1f} max={max(times):.1f}")
    print("decision sources:", dict(sources.most_common()))
    print("questions with errors:", errors)
    man = os.path.join(result_dir, "logs", "run_manifest.json")
    if os.path.exists(man):
        m = json.load(open(man, encoding="utf-8"))
        print(f"wall_s={m.get('wall_s')}  models={[x.get('name') for x in m.get('models', [])]}")
        warn = {b.get("business_id"): b.get("warnings") for b in m.get("businesses", []) if b.get("warnings")}
        if warn:
            print("business warnings:", json.dumps(warn, ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "/result"))
