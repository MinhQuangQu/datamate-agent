"""Accent-insensitive entity linker over the value index (05.4 §4)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz

from dma.ingest.parsers import UNITS
from dma.semantic.value_index import ValueIndex, is_common

UNIT_WORDS = set(UNITS) | {"bong", "lang", "canh", "to", "ly", "phan"}

STOPWORDS = {"la", "gi", "bao", "nhieu", "nao", "nhat", "cua", "cho", "trong", "thang", "tuan", "nam", "ban", "duoc",
             "dc", "co", "khong", "ko", "k", "hok", "mot", "cac", "nhung", "va", "voi", "vs", "o", "tai", "tu", "den",
             "toi", "nay", "roi", "truoc", "qua", "ngay", "hom", "sp", "san", "pham", "mon", "hang", "mat", "khach",
             "kh", "top", "may", "bn", "doanh", "thu", "so", "luong", "tien", "tong", "chay", "it", "nhieu", "cai",
             "bi", "da", "dang", "se", "the", "nhu", "ra", "vao", "len", "xuong", "het", "con", "lai", "ai", "dau",
             "tren", "duoi", "hon", "so", "sanh", "bang", "chiem", "phan", "tram", "ty", "le", "ti", "trung", "binh",
             "moi", "don", "hd", "di", "ve", "voi", "thi", "ma", "neu", "a", "ah", "nhe", "z", "vay", "j"}


@dataclass
class EntityCandidate:
    kind: str
    code: str | None
    name: str
    score: float
    freq: int = 0


@dataclass
class EntityMention:
    text: str
    start: int
    end: int                                  # token span [start, end)
    candidates: list[EntityCandidate] = field(default_factory=list)

    @property
    def best(self) -> EntityCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def score(self) -> float:
        return self.candidates[0].score if self.candidates else 0.0

    @property
    def margin(self) -> float:
        if len(self.candidates) < 2:
            return self.score
        return round(self.candidates[0].score - self.candidates[1].score, 4)

    def as_json(self) -> dict:
        return {"text": self.text, "span": [self.start, self.end], "score": self.score, "margin": self.margin,
                "candidates": [asdict(c) for c in self.candidates]}


def _score(m: str, mtoks: list[str], e, vi: ValueIndex) -> float:
    if e.code and m == e.code.lower():
        return 1.0
    folds = {e.fold} | {f for f in (vi_fold(v) for v in e.variants)}
    if m in folds:
        return 0.98
    if not set(mtoks) & set(e.tokens):
        return 0.0
    tsr = fuzz.token_set_ratio(m, e.fold) / 100.0
    pr = fuzz.partial_ratio(m, e.fold) / 100.0
    etoks = set(e.tokens)
    common = set(mtoks) & etoks
    cov = len(common) / max(1, len(set(e.fold.split())))
    precision = len(common) / max(1, len(set(mtoks)))        # extra words in the mention are not the entity
    s = (0.6 * tsr + 0.4 * pr) * (0.75 + 0.25 * min(1.0, cov))
    rare = sum(vi.idf.get(t, 0) for t in common if not is_common(t))
    tot = sum(vi.idf.get(t, 0) for t in set(e.fold.split())) or 1.0
    s += 0.1 * rare / tot
    if common and all(is_common(t) for t in common):
        s -= 0.15
    if len(mtoks) == 1 and len(e.fold.split()) >= 3:
        s -= 0.1
    return round(min(s, 0.97) * (0.6 + 0.4 * precision), 4)


def vi_fold(s: str) -> str:
    from dma.textnorm import fold
    return fold(s)


# business vocabulary of the questions: a token inside one of these phrases is a word of the question, not a
# (fuzzy) mention of a product — "tồn kho" must not link "kho" to "Khô mực nướng"; exact names still link
DOMAIN_PHRASES = ("ton kho", "trong kho", "kho con", "doanh thu", "doanh so", "cong no", "con no", "tra no", "no them",
                  "nhap hang", "phieu nhap", "gia von", "gia nhap", "gia ban", "ban chay", "don hang", "hoa don",
                  "khach hang", "khach le", "nha cung cap", "het han", "han dung", "ty le", "ti le", "tien hang",
                  "chiet khau", "giam gia", "thanh toan", "chuyen khoan", "tien mat", "ca sang", "ca chieu", "cuoi ngay")


def _domain_mask(toks: list[str]) -> set[int]:
    mask: set[int] = set()
    for ph in DOMAIN_PHRASES:
        p = ph.split()
        for i in range(len(toks) - len(p) + 1):
            if toks[i:i + len(p)] == p:
                mask.update(range(i, i + len(p)))
    return mask


def link_entities(folded: str, vi: ValueIndex, top_k: int = 5, min_score: float = 0.60, max_n: int = 8) -> list[EntityMention]:
    toks = folded.split()
    domain = _domain_mask(toks)
    found: list[tuple[float, int, int, list[EntityCandidate]]] = []
    for n in range(min(max_n, len(toks)), 0, -1):
        for i in range(0, len(toks) - n + 1):
            span = toks[i:i + n]
            if all(t in STOPWORDS or t in UNIT_WORDS for t in span):
                # only an exact multi-token name may consist of function words ("Hoa lẻ" -> hoa le)
                m = " ".join(span)
                if n >= 2 and vi.by_fold.get(m):
                    ex = [vi.entries[j] for j in vi.by_fold[m]]
                    found.append((0.98, i, i + n, [EntityCandidate(e.kind, e.code, e.name, 0.98, e.freq) for e in ex][:top_k], True))
                continue
            if span[0] in STOPWORDS and n > 1 and span[-1] in STOPWORDS:
                continue
            if all(j in domain for j in range(i, i + n)):
                ex = vi.by_fold.get(" ".join(span))
                if ex:
                    found.append((0.98, i, i + n, [EntityCandidate(vi.entries[j].kind, vi.entries[j].code,
                                                                   vi.entries[j].name, 0.98, vi.entries[j].freq)
                                                    for j in ex][:top_k], n >= 2))
                continue
            content = [t for t in span if t not in STOPWORDS] or span
            ids = vi.candidates(set(content))
            if not ids:
                continue
            m = " ".join(span)
            cands = []
            for idx in ids:
                e = vi.entries[idx]
                s = _score(m, span, e, vi)
                if e.kind.startswith("custom:"):
                    s -= 0.05
                if s >= min_score:
                    cands.append(EntityCandidate(e.kind, e.code, e.name, s, e.freq))
            if not cands:
                continue
            cands.sort(key=lambda c: (-c.score, -c.freq, c.code or c.name))
            best = vi.lookup_fold(cands[0].name)
            btoks = set(best[0].tokens) if best else set()
            full = n >= 2 and cands[0].score >= 0.85 and all(t in btoks for t in content)
            found.append((cands[0].score, i, i + n, cands[:top_k], full))
    # greedy, non-overlapping: multi-token spans fully inside one entity first (longest wins:
    # "bia tiger" beats category "bia"), then best score with a small bonus per token
    found.sort(key=lambda x: (-(x[4] * (x[2] - x[1])), -(x[0] + 0.04 * (x[2] - x[1])), x[1], x[2]))
    taken: list[tuple[int, int]] = []
    out: list[EntityMention] = []
    for s, a, b, cands, _full in found:
        if any(not (b <= x or a >= y) for x, y in taken):
            continue
        taken.append((a, b))
        out.append(EntityMention(" ".join(toks[a:b]), a, b, cands))
    out.sort(key=lambda m: m.start)
    return out
