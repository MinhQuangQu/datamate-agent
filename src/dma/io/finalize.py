"""Safety net executed by predict.sh after `dma.run` (even if it crashed):
every question id gets exactly one valid line; invalid/duplicate/unknown lines are dropped;
missing ids become {"id": ..., "type": "abstain"}. Written atomically.

    python -m dma.io.finalize --data /data --result /result
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

ANSWER_TYPES = {"number", "percent", "string", "date", "boolean", "list", "table", "unanswerable", "abstain"}
_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")


def find_questions(data_root: str, max_depth: int = 3) -> str:
    """Shallowest questions.jsonl under data_root (deterministic)."""
    best = None
    root_depth = data_root.rstrip(os.sep).count(os.sep)
    for cur, dirs, files in os.walk(data_root):
        dirs.sort()
        depth = cur.count(os.sep) - root_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if "questions.jsonl" in files:
            cand = (depth, os.path.join(cur, "questions.jsonl"))
            best = cand if best is None or cand < best else best
    if best is None:
        raise FileNotFoundError(f"questions.jsonl not found under {data_root}")
    return best[1]


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_record(rec: object) -> str | None:
    """None if valid, else the reason."""
    if not isinstance(rec, dict) or not isinstance(rec.get("id"), str) or not rec["id"]:
        return "missing id"
    t = rec.get("type")
    if t not in ANSWER_TYPES:
        return f"bad type {t!r}"
    a = rec.get("answer")
    if t in ("number", "percent"):
        return None if _num(a) else "answer must be a finite number"
    if t == "string":
        return None if isinstance(a, str) and a.strip() else "answer must be a non-empty string"
    if t == "date":
        return None if isinstance(a, str) and _DATE.match(a) else "date must be yyyy-mm-dd or yyyy-mm"
    if t == "boolean":
        return None if isinstance(a, bool) else "answer must be true/false"
    if t == "list":
        ok = isinstance(a, list) and a and all(isinstance(x, str) or _num(x) for x in a)
        return None if ok else "list must be non-empty, of strings/numbers"
    if t == "table":
        if not isinstance(a, dict):
            return "table must be an object"
        cols, rows = a.get("columns"), a.get("rows")
        if not (isinstance(cols, list) and cols and all(isinstance(c, str) for c in cols)):
            return "table.columns must be non-empty strings"
        if not (isinstance(rows, list) and all(isinstance(r, list) and len(r) == len(cols) for r in rows)):
            return "table.rows must be lists with len(columns) cells"
        return None
    if "answer" in rec and rec["answer"] is not None:
        return f"{t} must not carry an answer"
    if "reason" in rec and not isinstance(rec["reason"], str):
        return "reason must be a string"
    return None


def read_ids(questions_path: str) -> list[str]:
    ids, seen = [], set()
    with open(questions_path, encoding="utf-8-sig") as f:
        for line in f:
            try:
                qid = json.loads(line)["id"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            if qid not in seen:
                seen.add(qid)
                ids.append(qid)
    return ids


def finalize(data_root: str, result_dir: str) -> dict:
    ids = read_ids(find_questions(data_root))
    wanted = set(ids)
    sub = os.path.join(result_dir, "submission.jsonl")
    kept: dict[str, dict] = {}
    stats = {"questions": len(ids), "invalid": 0, "duplicate": 0, "unknown_id": 0}
    if os.path.exists(sub):
        with open(sub, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid"] += 1
                    continue
                if validate_record(rec) is not None:
                    stats["invalid"] += 1
                    continue
                if rec["id"] not in wanted:
                    stats["unknown_id"] += 1
                elif rec["id"] in kept:
                    stats["duplicate"] += 1
                else:
                    kept[rec["id"]] = rec
    stats["answered"] = sum(kept[i]["type"] != "abstain" for i in kept)
    stats["filled_abstain"] = len(ids) - len(kept)
    os.makedirs(result_dir, exist_ok=True)
    tmp = sub + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for qid in ids:
            rec = kept.get(qid, {"id": qid, "type": "abstain"})
            f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, sub)
    stats["bytes"] = os.path.getsize(sub)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--result", default="/result")
    a = ap.parse_args(argv)
    try:
        stats = finalize(a.data, a.result)
    except FileNotFoundError as e:
        print(f"[finalize] {e}", file=sys.stderr)
        return 1
    print("[finalize]", json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
