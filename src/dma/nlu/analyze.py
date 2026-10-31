"""analyze_question(): deterministic NLU, no model (05.4 §1, §8)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from dma.ingest.parsers import parse_number
from dma.nlu.concepts import CONCEPT_GAP, detect_concepts, detect_k, ranking_direction, type_cues
from dma.nlu.link import EntityMention, link_entities
from dma.nlu.typos import build_lexicon, fix_telex_text, retype_telex
from dma.textnorm import expand_abbrev, fold, has_vietnamese_diacritics, nfc, strip_accents
from dma.timeparse.resolver import Period, TimeParse, resolve

WEEKDAY_VI = {1: "Thứ Hai", 2: "Thứ Ba", 3: "Thứ Tư", 4: "Thứ Năm", 5: "Thứ Sáu", 6: "Thứ Bảy", 7: "Chủ nhật"}
_NUM = re.compile(r"(?<![\w/])(\d+(?:[.,]\d+)?\s*(?:k|tr|trieu|triệu|nghin|nghìn|ngan|ngàn|ty|tỷ|%)?(?:\d+)?)(?![\w/])", re.I)


@dataclass
class NumberMention:
    text: str
    value: float
    is_percent: bool = False


@dataclass
class QuestionAnalysis:
    raw: str
    as_of: date
    nfc: str
    folded: str
    expanded: str
    abbrev_hints: list[str]
    has_diacritics: bool
    time: TimeParse
    numbers: list[NumberMention]
    entities: list[EntityMention]
    concepts: list[str]
    gaps_hit: list[str]
    type_cues: list[tuple[str, float]]
    k: int | None
    ranking_direction: str | None
    template_key: str
    restored: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def strong_type(self) -> str | None:
        return self.type_cues[0][0] if self.type_cues and self.type_cues[0][1] >= 0.8 else None

    def periods_json(self) -> list[dict]:
        return [p.as_json() for p in self.time.periods]

    def entity_rows(self, k: int = 3) -> list[dict]:
        rows = []
        for m in self.entities:
            for c in m.candidates[:k]:
                rows.append({"mention": m.text, "kind": c.kind, "code": c.code, "name": c.name, "score": c.score})
        return rows

    def as_json(self) -> dict:
        return {"normalized": self.expanded, "folded": self.folded, "abbrev_hints": self.abbrev_hints,
                "time": self.time.as_json(), "entities": [m.as_json() for m in self.entities],
                "numbers": [n.__dict__ for n in self.numbers], "concepts": self.concepts, "gaps_hit": self.gaps_hit,
                "type_cues": self.type_cues, "k": self.k, "ranking_direction": self.ranking_direction,
                "template_key": self.template_key}


def _numbers(text: str) -> list[NumberMention]:
    out = []
    for m in _NUM.finditer(nfc(text)):
        p = parse_number(m.group(1).strip())
        if p is not None:
            out.append(NumberMention(m.group(1).strip(), p.value * (1 if not p.multiplier else 1), p.is_percent))
    return out


def lexicon_for(vindex) -> set[str]:
    """Known folded words for typo repair: core lexicon + entity-name tokens (+ README/rulebook, set by the BKP)."""
    if vindex is None:
        return build_lexicon()
    lex = getattr(vindex, "_lexicon", None)
    if lex is None:
        lex = build_lexicon(extra_tokens={t for e in vindex.entries for t in e.tokens})
        lex |= getattr(vindex, "_extra_words", set())
        vindex._lexicon = lex
    return lex


def analyze_question(question: str, as_of: date, vindex, gaps: list[str], time_rules, calendar,
                     coverage: tuple[str, str] | None = None) -> QuestionAnalysis:
    raw = question
    n = nfc(question).strip()
    lex = lexicon_for(vindex)
    plain, typo_fixes = fix_telex_text(strip_accents(n.replace("%", " phần trăm ")), lex)
    folded = fold(plain)                      # fold() drops symbols; "%" carries the percent cue
    expanded, hints = expand_abbrev(n)
    if typo_fixes:
        expanded = retype_telex(expanded, typo_fixes)
    tp = resolve(plain, as_of, time_rules, calendar)
    if "all_time" in tp.flags and coverage:
        start = date.fromisoformat(coverage[0][:10])
        tp.periods.insert(0, Period(min(start, as_of), as_of, "all_time", "tu truoc toi nay"))
    ents = link_entities(folded, vindex) if vindex is not None else []
    concepts = detect_concepts(folded)
    gap_set = set(gaps or [])
    gaps_hit = sorted({CONCEPT_GAP[c] for c in concepts if c in CONCEPT_GAP} & gap_set)
    if "forecast" in concepts:
        gaps_hit.append("forecast")
    if coverage and tp.periods:
        # rulebook §7: a period lying ENTIRELY outside the shop's data range is unanswerable (not 0)
        c0, c1 = date.fromisoformat(coverage[0][:10]), date.fromisoformat(coverage[1][:10])
        asked = [p for p in tp.periods if p.label != "all_time"]
        outside = [p.end < c0 or p.start > min(c1, as_of) for p in asked]
        # a comparison needs BOTH sides: "tháng 2/2025 so với tháng liền trước" with no January data is unanswerable
        compares = len(asked) >= 2 and re.search(r"so (voi|vs)|tang hay giam|thay doi|chenh lech|cung ky", folded)
        if asked and (all(outside) or (compares and any(outside))):
            gaps_hit.append("out_of_coverage")
    if any(p.start > as_of for p in tp.periods):
        if "future_period" not in tp.flags:
            tp.flags.append("future_period")
    cues = type_cues(folded)
    k = detect_k(folded)
    direction = ranking_direction(folded)
    # template key: entities -> <kind>, time spans -> <time:label>, numbers -> <num>
    toks = folded.split()
    for m in sorted(ents, key=lambda m: -m.start):
        kind = m.best.kind.split(":")[0] if m.best else "entity"
        toks[m.start:m.end] = [f"<{kind}>"]
    key = " ".join(toks)
    for p in tp.periods:
        if p.text:
            key = key.replace(fold(p.text), f"<time:{re.sub(r'_\d{4}.*$', '', p.label)}>", 1)
    key = re.sub(r"\b\d+\b", "<num>", key)
    return QuestionAnalysis(raw=raw, as_of=as_of, nfc=n, folded=folded, expanded=expanded, abbrev_hints=hints,
                            has_diacritics=has_vietnamese_diacritics(n), time=tp, numbers=_numbers(n), entities=ents,
                            concepts=concepts, gaps_hit=gaps_hit, type_cues=cues, k=k, ranking_direction=direction,
                            template_key=key, extra={"typo_fixes": typo_fixes} if typo_fixes else {})


def weekday_vi(d: date, names: dict[int, str] | None = None) -> str:
    return (names or WEEKDAY_VI)[d.isoweekday()]


_ = timedelta
