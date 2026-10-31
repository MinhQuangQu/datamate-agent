"""Local replica of the DataMate official scorer — Penalized Accuracy (PA).

    PA = (#correct - #wrong) / N        abstain or missing -> 0

A prediction is correct only if its `type` equals the reference type AND the value
matches the rule of that type. Rules follow Overview -> Evaluation of the portal.
Ambiguous corners of the official implementation are exposed as options so the
team can test both readings (see ScoreOptions).

CLI:
    python -m dma.eval.scorer --pred submission.jsonl --gold answers.jsonl \
        [--questions questions.jsonl] [--report report.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from dma.textnorm import scorer_norm

ANSWER_TYPES = ("number", "percent", "string", "date", "boolean", "list", "table", "unanswerable")
ALL_TYPES = ANSWER_TYPES + ("abstain",)


@dataclass
class ScoreOptions:
    # number: do NOT round the prediction (forces our formatter to round). The
    # reference is already an int or rounded to 2 decimals.
    round_number_pred: bool = False
    # table rows compared as multiset (stricter) instead of set.
    table_rows_multiset: bool = True
    # accept numbers given as numeric strings ("123")? Official behaviour unknown -> strict.
    accept_numeric_strings: bool = False


@dataclass
class ItemResult:
    id: str
    score: int
    reason: str
    gold_type: str | None = None
    pred_type: str | None = None


@dataclass
class Report:
    n: int = 0
    correct: int = 0
    wrong: int = 0
    abstain: int = 0
    items: list[ItemResult] = field(default_factory=list)

    @property
    def pa(self) -> float:
        return (self.correct - self.wrong) / self.n if self.n else 0.0


# ---------------------------------------------------------------- primitives
def _is_number(x: Any, opt: ScoreOptions) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return math.isfinite(x)
    if opt.accept_numeric_strings and isinstance(x, str):
        try:
            return math.isfinite(float(x))
        except ValueError:
            return False
    return False


def _num(x: Any) -> float:
    return float(x)


def match_number(pred: Any, gold: Any, opt: ScoreOptions) -> bool:
    if not (_is_number(pred, opt) and _is_number(gold, ScoreOptions(accept_numeric_strings=True))):
        return False
    p, g = _num(pred), _num(gold)
    if opt.round_number_pred:
        p = round(p, 2)
    return p == g or abs(p - g) <= 1e-9 * max(1.0, abs(g))


def match_percent(pred: Any, gold: Any, opt: ScoreOptions) -> bool:
    if not (_is_number(pred, opt) and _is_number(gold, ScoreOptions(accept_numeric_strings=True))):
        return False
    return round(_num(pred), 2) == round(_num(gold), 2)


def match_string(pred: Any, gold: Any, aliases: list[Any] | None) -> bool:
    if not isinstance(pred, str):
        return False
    accepted = {scorer_norm(gold)} | {scorer_norm(a) for a in (aliases or []) if isinstance(a, (str, int, float))}
    return scorer_norm(pred) in accepted


def match_date(pred: Any, gold: Any) -> bool:
    return isinstance(pred, str) and isinstance(gold, str) and pred.strip() == gold.strip()


def match_boolean(pred: Any, gold: Any) -> bool:
    return isinstance(pred, bool) and isinstance(gold, bool) and pred == gold


def _element_aliases(aliases: Any, i: int, n: int) -> list[Any] | None:
    """Per-position aliases if alias is a list aligned with the answer list."""
    if isinstance(aliases, list) and len(aliases) == n and all(isinstance(a, list) for a in aliases):
        return aliases[i]
    return None


def match_list(pred: Any, gold: Any, aliases: Any, opt: ScoreOptions) -> bool:
    if not (isinstance(pred, list) and isinstance(gold, list)) or len(pred) != len(gold):
        return False
    for i, (p, g) in enumerate(zip(pred, gold)):
        if _is_number(g, ScoreOptions()):
            if not match_number(p, g, opt):
                return False
        elif not match_string(p, g, _element_aliases(aliases, i, len(gold))):
            return False
    return True


def _cell_key(v: Any, numeric: bool) -> Any:
    if numeric:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return ("bad", repr(v))
        return ("n", round(float(v), 9))
    return ("s", scorer_norm(v))


def match_table(pred: Any, gold: Any, opt: ScoreOptions) -> bool:
    if not (isinstance(pred, dict) and isinstance(gold, dict)):
        return False
    pc, pr, gc, gr = pred.get("columns"), pred.get("rows"), gold.get("columns"), gold.get("rows")
    if not all(isinstance(x, list) for x in (pc, pr, gc, gr)):
        return False
    if len(pc) != len(gc) or len(pr) != len(gr):
        return False
    pn, gn = [scorer_norm(c) for c in pc], [scorer_norm(c) for c in gc]
    if sorted(pn) == sorted(gn) and len(set(gn)) == len(gn):
        perm = [pn.index(c) for c in gn]          # match columns by normalized name
    else:
        perm = list(range(len(gc)))               # fall back to position
    numeric_col = [all(_is_number(row[j], ScoreOptions()) for row in gr if len(row) == len(gc)) and len(gr) > 0
                   for j in range(len(gc))]
    try:
        p_keys = [tuple(_cell_key(row[perm[j]], numeric_col[j]) for j in range(len(gc))) for row in pr]
        g_keys = [tuple(_cell_key(row[j], numeric_col[j]) for j in range(len(gc))) for row in gr]
    except (IndexError, TypeError):
        return False
    if opt.table_rows_multiset:
        return Counter(p_keys) == Counter(g_keys)
    return set(p_keys) == set(g_keys)


# ---------------------------------------------------------------- one item
def score_item(pred: dict | None, gold: dict, opt: ScoreOptions | None = None) -> tuple[int, str]:
    opt = opt or ScoreOptions()
    if pred is None:
        return 0, "missing"
    pt, gt = pred.get("type"), gold.get("type")
    if pt == "abstain":
        return 0, "abstain"
    if pt not in ANSWER_TYPES:
        return -1, "invalid_type"
    if pt != gt:
        return -1, f"type_mismatch:{pt}->{gt}"
    if gt == "unanswerable":
        return 1, "ok"
    p, g, al = pred.get("answer"), gold.get("answer"), gold.get("alias")
    ok = {
        "number": lambda: match_number(p, g, opt),
        "percent": lambda: match_percent(p, g, opt),
        "string": lambda: match_string(p, g, al if isinstance(al, list) else None),
        "date": lambda: match_date(p, g),
        "boolean": lambda: match_boolean(p, g),
        "list": lambda: match_list(p, g, al, opt),
        "table": lambda: match_table(p, g, opt),
    }[gt]()
    return (1, "ok") if ok else (-1, "value_mismatch")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def score(preds: list[dict], golds: list[dict], opt: ScoreOptions | None = None) -> Report:
    by_id: dict[str, dict] = {}
    dup = 0
    for p in preds:
        if p.get("id") in by_id:
            dup += 1
            continue                              # keep first occurrence
        by_id[p.get("id")] = p
    if dup:
        print(f"[scorer] WARNING: {dup} duplicated ids in predictions (kept first)", file=sys.stderr)
    rep = Report()
    for g in golds:
        p = by_id.get(g["id"])
        s, why = score_item(p, g, opt)
        rep.n += 1
        rep.correct += s == 1
        rep.wrong += s == -1
        rep.abstain += s == 0
        rep.items.append(ItemResult(g["id"], s, why, g.get("type"), p.get("type") if p else None))
    return rep


def breakdown(rep: Report, key_of: dict[str, str] | None = None) -> dict[str, dict]:
    """Group results by gold type (default) or by a custom key (e.g. business_id)."""
    groups: dict[str, list[ItemResult]] = defaultdict(list)
    for it in rep.items:
        groups[key_of.get(it.id, "?") if key_of else it.gold_type].append(it)
    out = {}
    for k, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        c = sum(i.score == 1 for i in items)
        w = sum(i.score == -1 for i in items)
        out[str(k)] = {"n": len(items), "correct": c, "wrong": w, "abstain": len(items) - c - w,
                       "pa": round((c - w) / len(items), 4)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--questions", help="questions.jsonl, enables per-business breakdown")
    ap.add_argument("--report", help="write a JSON report here")
    ap.add_argument("--round-number-pred", action="store_true")
    ap.add_argument("--rows-as-set", action="store_true")
    a = ap.parse_args(argv)
    opt = ScoreOptions(round_number_pred=a.round_number_pred, table_rows_multiset=not a.rows_as_set)
    rep = score(read_jsonl(a.pred), read_jsonl(a.gold), opt)
    print(f"N={rep.n}  correct={rep.correct}  wrong={rep.wrong}  abstain={rep.abstain}  PA={rep.pa:.4f}")
    by_type = breakdown(rep)
    for k, v in by_type.items():
        print(f"  {k:<13} n={v['n']:<5} PA={v['pa']:+.3f}  ok={v['correct']} wrong={v['wrong']} abst={v['abstain']}")
    reasons = Counter(i.reason for i in rep.items if i.score == -1)
    if reasons:
        print("  wrong reasons:", dict(reasons.most_common(10)))
    if a.report:
        biz = None
        if a.questions:
            biz = {q["id"]: q.get("business_id", "?") for q in read_jsonl(a.questions)}
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump({"pa": rep.pa, "n": rep.n, "correct": rep.correct, "wrong": rep.wrong,
                       "abstain": rep.abstain, "by_type": by_type,
                       "by_business": breakdown(rep, biz) if biz else None,
                       "items": [i.__dict__ for i in rep.items]}, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
