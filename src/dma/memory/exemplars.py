"""Exemplar memory (07 §3): verified (question -> DMQL / code) pairs, retrieved by BM25 on
template keys + small bonus for shared concepts / type cue. Deterministic (ties -> ex_id),
diverse (<= max_per_template per template), and filterable by business (LOBO CV)."""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class Exemplar:
    ex_id: str
    question_norm: str
    template_key: str
    answer_type: str
    op: str | None = None
    dmql: dict | None = None
    code: str | None = None
    concepts: list[str] = field(default_factory=list)
    business_type: str | None = None
    source: str = "seed"
    business_id: str | None = None          # only used to EXCLUDE during CV, never for logic
    verified: bool = True


def _toks(s: str) -> list[str]:
    return [t for t in s.replace("<", " <").replace(">", "> ").split() if t]


class ExemplarMemory:
    def __init__(self, items: list[Exemplar], k1: float = 1.2, b: float = 0.75):
        self.items = sorted(items, key=lambda e: e.ex_id)
        self.k1, self.b = k1, b
        self.docs = [_toks(e.template_key) for e in self.items]
        self.avgdl = sum(map(len, self.docs)) / max(1, len(self.docs))
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    @classmethod
    def load(cls, path: str | None) -> "ExemplarMemory":
        items = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        items.append(Exemplar(**{k: v for k, v in d.items() if k in Exemplar.__dataclass_fields__}))
        return cls(items)

    def __len__(self) -> int:
        return len(self.items)

    def bm25(self, q: list[str], i: int) -> float:
        tf, dl = self.tf[i], len(self.docs[i])
        s = 0.0
        for t in q:
            if t in tf:
                f = tf[t]
                s += self.idf.get(t, 0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl)))
        return s

    def retrieve(self, template_key: str, concepts: list[str], type_cue: str | None, k: int = 3,
                 max_per_template: int = 1, exclude_businesses: set[str] | None = None, shift: int = 0,
                 need: str = "dmql") -> list[Exemplar]:
        q = _toks(template_key)
        cs = set(concepts)
        scored = []
        for i, e in enumerate(self.items):
            if exclude_businesses and e.business_id in exclude_businesses:
                continue
            if need == "dmql" and not e.dmql:
                continue
            if need == "code" and not e.code:
                continue
            s = self.bm25(q, i) + 0.5 * len(cs & set(e.concepts)) + (0.7 if type_cue and e.answer_type == type_cue else 0.0)
            scored.append((-s, e.ex_id, e))
        scored.sort(key=lambda x: (x[0], x[1]))
        out, per = [], Counter()
        pool = [e for _, _, e in scored]
        if shift:                       # extra samples see different exemplars (05.7 §4)
            pool = pool[shift:] + pool[:shift]
        for e in pool:
            if per[e.template_key] >= max_per_template:
                continue
            per[e.template_key] += 1
            out.append(e)
            if len(out) >= k:
                break
        return out


def render_dmql_examples(exs: list[Exemplar]) -> str:
    if not exs:
        return "(none)"
    return "\n".join(f'Q: "{e.question_norm}"\n{json.dumps(e.dmql, ensure_ascii=False)}' for e in exs)


def render_code_examples(exs: list[Exemplar]) -> str:
    if not exs:
        return "(none)"
    return "\n".join(f'Q: "{e.question_norm}"\n```python\n{e.code}\n```' for e in exs)
