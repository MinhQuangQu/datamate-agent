"""SplitContext: everything shared by all businesses of a split (05.3, 04 §5)."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from dma.output.formatter import FormatConfig
from dma.rules.rulespec import RuleSpec, render_digest, time_rules_from
from dma.timeparse.calendar import LunarCalendar
from dma.timeparse.resolver import TimeRules


@dataclass
class SplitContext:
    rulespec: RuleSpec
    calendar: LunarCalendar
    time_rules: TimeRules
    digest: str
    weekday_names: dict[int, str]
    format_cfg: FormatConfig
    rulebook_text: str = ""
    report: dict = field(default_factory=dict)
    calendar_path: str | None = None
    dir: str = ""

    @classmethod
    def make(cls, rulespec: RuleSpec, calendar: LunarCalendar, weekday_names: dict[int, str], format_cfg: FormatConfig,
             rulebook_text: str = "", report: dict | None = None, calendar_path: str | None = None) -> "SplitContext":
        digest = render_digest(rulespec)
        if rulebook_text and rulebook_text.strip():
            digest += ("\n\n## RULEBOOK (verbatim, authoritative; the digest above is a summary of it)\n"
                       + rulebook_text.strip()[:20000])
        return cls(rulespec, calendar, time_rules_from(rulespec.time), digest, weekday_names, format_cfg,
                   rulebook_text, report or {}, calendar_path)

    def digest_for(self, question: str, mode: str = "full", k: int = 14) -> str:
        """Rulebook context of one question. "full": structured digest + whole rulebook (default);
        "relevant": structured digest + the k rulebook rows closest to the question (hosted APIs with small
        per-minute token budgets, faster prefill); the agent can still read any rule with dm.rules_text()."""
        if mode != "relevant" or not self.rulebook_text.strip():
            return self.digest
        import math
        import re as _re
        from dma.rules.parse import parse_rulebook
        from dma.textnorm import fold
        items = [it for it in parse_rulebook(self.rulebook_text) if it.text.strip()]
        docs = [set(fold(it.section + " " + it.text).split()) for it in items]
        df: dict[str, int] = {}
        for d in docs:
            for t in d:
                df[t] = df.get(t, 0) + 1
        q = {t for t in fold(question).split() if len(t) > 1 and not _re.fullmatch(r"\d+", t)}
        n = len(docs)
        scored = sorted(((sum(math.log(1 + n / df[t]) for t in q & d), i) for i, d in enumerate(docs)),
                        key=lambda x: (-x[0], x[1]))
        keep = sorted(i for sc, i in scored[:k] if sc > 0)
        rows = "\n".join(f"- [{items[i].section}] {items[i].text}" for i in keep)
        return (render_digest(self.rulespec) + "\n\n## RULEBOOK ROWS RELEVANT TO THIS QUESTION (verbatim; "
                "authoritative; the full rulebook is searchable with dm.rules_text)\n" + rows)

    def save(self, d: str) -> None:
        os.makedirs(d, exist_ok=True)
        self.dir = d
        with open(os.path.join(d, "rulespec.json"), "w", encoding="utf-8") as f:
            f.write(self.rulespec.model_dump_json(indent=1))
        with open(os.path.join(d, "rule_digest.md"), "w", encoding="utf-8") as f:
            f.write(self.digest)
        with open(os.path.join(d, "rulebook.md"), "w", encoding="utf-8") as f:
            f.write(self.rulebook_text or "")
        with open(os.path.join(d, "split_ctx.json"), "w", encoding="utf-8") as f:
            json.dump({"calendar_path": self.calendar_path, "weekday_names": {str(k): v for k, v in self.weekday_names.items()},
                       "format": {"rounding": self.format_cfg.rounding, "max_list_len": self.format_cfg.max_list_len,
                                  "max_table_rows": self.format_cfg.max_table_rows},
                       "report": self.report, "calendar_report": self.calendar.report.__dict__},
                      f, ensure_ascii=False, indent=1, default=str)

    @classmethod
    def load(cls, d: str) -> "SplitContext":
        with open(os.path.join(d, "split_ctx.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(d, "rulespec.json"), encoding="utf-8") as f:
            rs = RuleSpec.model_validate_json(f.read())
        cal = load_calendar(meta.get("calendar_path"))[0]
        rb = open(os.path.join(d, "rulebook.md"), encoding="utf-8").read() if os.path.exists(os.path.join(d, "rulebook.md")) else ""
        ctx = cls.make(rs, cal, {int(k): v for k, v in meta["weekday_names"].items()}, FormatConfig(**meta["format"]),
                       rb, meta.get("report"), meta.get("calendar_path"))
        ctx.dir = d
        return ctx


def load_calendar(path: str | None) -> tuple[LunarCalendar, list[str]]:
    warns = []
    if path and os.path.exists(path):
        try:
            cal = LunarCalendar.from_csv(path)
            warns += cal.report.warnings
            return cal, warns
        except Exception as e:  # noqa: BLE001
            warns.append(f"calendar.csv unparseable ({e}) -> algorithm")
    return LunarCalendar.from_algorithm(), warns
