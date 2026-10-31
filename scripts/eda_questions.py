"""Day-1 EDA #2 — questions & answers: types, templates, formats, unanswerable patterns.

    python scripts/eda_questions.py /path/to/train --out reports/questions

Answers only exist for train. Never feed public/private answers into anything
(they do not exist anyway) and never train on public/private questions (Rule 6).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.textnorm import UNAMBIGUOUS, AMBIGUOUS, fold, has_vietnamese_diacritics  # noqa: E402


def read_jsonl(p: str) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


_NUM = re.compile(r"\b\d+([.,]\d+)*\s*(k|tr|trieu|nghin|%)?\b")
_DATE = re.compile(r"\b\d{1,2}\s*[/-]\s*\d{1,2}(\s*[/-]\s*\d{2,4})?\b")


def template_of(q: str, entity_words: list[str]) -> str:
    s = fold(q)
    for w in sorted(entity_words, key=len, reverse=True):
        if len(w) >= 4 and w in s:
            s = s.replace(w, " <ent> ")
    s = _DATE.sub(" <date> ", s)
    s = _NUM.sub(" <num> ", s)
    return re.sub(r"\s+", " ", s).strip()


def decimals(x: float) -> int:
    s = repr(float(x))
    return 0 if float(x).is_integer() else len(s.split(".")[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split_dir")
    ap.add_argument("--out", default="reports/questions")
    a = ap.parse_args()
    qs = read_jsonl(os.path.join(a.split_dir, "questions.jsonl"))
    ans_path = os.path.join(a.split_dir, "answers.jsonl")
    ans = {r["id"]: r for r in read_jsonl(ans_path)} if os.path.exists(ans_path) else {}
    rep: dict = {"n_questions": len(qs), "n_answers": len(ans)}
    rep["per_business"] = Counter(q["business_id"] for q in qs).most_common()
    rep["as_of_range"] = [min(q["as_of_date"] for q in qs), max(q["as_of_date"] for q in qs)]
    rep["no_diacritics_rate"] = round(sum(not has_vietnamese_diacritics(q["question"]) for q in qs) / len(qs), 3)
    toks = Counter(t for q in qs for t in re.findall(r"\w+", q["question"].lower()))
    rep["abbrev_hits"] = {k: toks[k] for k in list(UNAMBIGUOUS) + list(AMBIGUOUS) if toks.get(k)}
    rep["top_tokens"] = [t for t, _ in Counter({fold(k): v for k, v in toks.items()}).most_common(120)]
    if ans:
        types = Counter(r["type"] for r in ans.values())
        rep["answer_types"] = dict(types)
        rep["unanswerable_rate"] = round(types.get("unanswerable", 0) / len(ans), 4)
        nums = [r["answer"] for r in ans.values() if r["type"] == "number"]
        rep["number_decimals"] = dict(Counter(decimals(x) for x in nums))
        pcts = [r["answer"] for r in ans.values() if r["type"] == "percent"]
        rep["percent_negative_rate"] = round(sum(p < 0 for p in pcts) / max(1, len(pcts)), 3)
        rep["percent_abs_range"] = [min(map(abs, pcts), default=None), max(map(abs, pcts), default=None)]
        rep["date_formats"] = dict(Counter("yyyy-mm" if len(r["answer"]) == 7 else "yyyy-mm-dd"
                                           for r in ans.values() if r["type"] == "date"))
        rep["list_lengths"] = dict(Counter(len(r["answer"]) for r in ans.values() if r["type"] == "list"))
        rep["table_columns"] = Counter(tuple(r["answer"]["columns"]) for r in ans.values()
                                       if r["type"] == "table").most_common(40)
        rep["alias_shapes"] = dict(Counter(type(r.get("alias")).__name__ + (
            "[list]" if isinstance(r.get("alias"), list) and r["alias"] and isinstance(r["alias"][0], list) else "")
            for r in ans.values()))
        weekdayish = [r["answer"] for r in ans.values() if r["type"] == "string" and re.match(r"(?i)^(thứ|chủ nhật|t\d|cn)", str(r["answer"]))]
        rep["weekday_answer_forms"] = dict(Counter(weekdayish))
        rep["string_answer_examples"] = [r["answer"] for r in ans.values() if r["type"] == "string"][:40]
        # templates
        tmpl: dict[str, Counter] = defaultdict(Counter)
        examples: dict[str, str] = {}
        for q in qs:
            r = ans.get(q["id"])
            if not r:
                continue
            ent = []
            vals = r.get("answer")
            for v in (vals if isinstance(vals, list) else [vals]):
                if isinstance(v, str):
                    ent.append(fold(v))
            for al in r.get("alias") or []:
                if isinstance(al, str):
                    ent.append(fold(al))
            t = template_of(q["question"], ent)
            tmpl[t][r["type"]] += 1
            examples.setdefault(t, q["question"])
        top = sorted(tmpl.items(), key=lambda kv: -sum(kv[1].values()))
        rep["n_templates"] = len(top)
        rep["top_templates"] = [{"template": t, "n": sum(c.values()), "types": dict(c), "example": examples[t]}
                                for t, c in top[:150]]
        rep["unanswerable_examples"] = [{"q": q["question"], "reason": ans[q["id"]].get("reason")}
                                        for q in qs if ans.get(q["id"], {}).get("type") == "unanswerable"][:60]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "questions_eda.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("top_templates", "top_tokens",
                      "string_answer_examples", "unanswerable_examples", "table_columns")},
                     ensure_ascii=False, indent=1, default=str))
    print(f"\n{rep.get('n_templates', '?')} templates; full report in {a.out}/questions_eda.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
