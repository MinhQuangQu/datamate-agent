"""Mine verified exemplars from CV runs (07 §3.1): for every question whose plan or code
candidate matched the gold answer, keep the shortest correct DMQL plan and code trajectory,
with dates / entity values abstracted to placeholders (structure, not values).

    python scripts/mine_exemplars.py --runs runs/1105_lobo runs/1106_lobo --split data/raw/train \
        --out assets/exemplars.jsonl [--keep-seed]
Run CV with a higher-temperature profile to get more diverse samples (rejection sampling).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.eval.scorer import read_jsonl, score_item  # noqa: E402
from dma.eval.data import answers_path  # noqa: E402
from dma.io.split import discover_split  # noqa: E402


def abstract_plan(plan: dict, nlu: dict) -> dict:
    periods = (nlu.get("time") or {}).get("periods", [])
    s = json.dumps(plan, ensure_ascii=False)
    for p in periods:
        lab = re.sub(r"_\d{4}.*$", "", p["label"])
        s = s.replace(f'"{p["start"]}"', f'"<time:{lab}.start>"').replace(f'"{p["end"]}"', f'"<time:{lab}.end>"')
    s = re.sub(r'"\d{4}-\d{2}-\d{2}"', '"<date>"', s)
    d = json.loads(s)
    kinds: dict[str, int] = {}
    for f in d.get("filters", []):
        if f.get("field") in ("product", "customer", "category", "branch", "staff", "supplier", "channel", "payment_method"):
            kinds[f["field"]] = kinds.get(f["field"], 0) + 1
            f["value"] = f"<{f['field']}:{kinds[f['field']]}>"
    return d


def abstract_code(code: str, nlu: dict) -> str:
    for m in nlu.get("entities", []):
        for c in m.get("candidates", [])[:3]:
            for v in (c.get("code"), c.get("name")):
                if v and len(v) >= 3:
                    code = code.replace(repr(v), repr(f"<{c['kind']}:1>")).replace(f'"{v}"', f'"<{c["kind"]}:1>"')
    return code


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="assets/exemplars.jsonl")
    ap.add_argument("--keep-seed", action="store_true")
    a = ap.parse_args(argv)
    sp = discover_split(a.split)
    gold = {g["id"]: g for g in read_jsonl(answers_path(a.split))}
    best: dict[str, dict] = {}
    for run in a.runs:
        for path in glob.glob(os.path.join(run, "**", "logs", "*.json"), recursive=True):
            if os.path.basename(path) in ("run_manifest.json",):
                continue
            try:
                tr = json.load(open(path, encoding="utf-8"))
            except ValueError:
                continue
            g = gold.get(tr.get("id"))
            if not g or "nlu" not in tr:
                continue
            nlu = tr["nlu"]
            for c in tr.get("candidates", []):
                rec = {"id": g["id"], "type": c["answer_type"]}
                if c["answer_type"] != "unanswerable":
                    rec["answer"] = c["value"]
                if score_item(rec, g)[0] != 1:
                    continue
                cur = best.setdefault(g["id"], {"q": tr, "plan": None, "code": None, "steps": 99})
                ev = c.get("evidence") or {}
                if c["solver"].startswith("plan") and ev.get("plan") and not c.get("repaired"):
                    cur["plan"] = cur["plan"] or ev["plan"]
                if c["solver"].startswith("code") and ev.get("steps"):
                    cells = [s["code"] for s in ev["steps"] if s.get("code")]
                    if cells and len(cells) < cur["steps"]:
                        cur["code"], cur["steps"] = "\n".join(cells), len(cells)
    out, seen_tk = [], {}
    if a.keep_seed and os.path.exists(a.out):
        out = [json.loads(line) for line in open(a.out, encoding="utf-8") if json.loads(line).get("source") == "seed"]
    for qid in sorted(best):
        b = best[qid]
        tr, nlu = b["q"], b["q"]["nlu"]
        tk = nlu.get("template_key", "")
        if seen_tk.get(tk, 0) >= 3:            # cap near-duplicates per template
            continue
        seen_tk[tk] = seen_tk.get(tk, 0) + 1
        out.append({"ex_id": f"train_{len(out):05d}", "source": "train", "business_id": tr.get("business_id"),
                    "question_norm": nlu.get("folded", ""), "template_key": tk, "concepts": nlu.get("concepts", []),
                    "answer_type": gold[qid]["type"], "op": (b["plan"] or {}).get("op"),
                    "dmql": abstract_plan(b["plan"], nlu) if b["plan"] else None,
                    "code": abstract_code(b["code"], nlu) if b["code"] else None, "verified": True})
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mine] {len(out)} exemplars -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
