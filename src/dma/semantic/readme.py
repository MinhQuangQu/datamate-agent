"""README understanding (deterministic part): data dictionary, unit conversions, coverage.

Shops document their files in README.md ("## Cột trong `file`, sheet `S`" + "- **Col**: description",
or "`table.col` là description"). The descriptions are matched to ontology roles through the
`readme_desc` phrases of assets/ontology.yaml (data, not code); the README text itself is also
given verbatim to the LLM (BSM refine, Business Card) for everything this parser does not cover.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from dma.textnorm import fold


@dataclass
class DictEntry:
    file: str | None          # file stem or table name the section refers to (folded), None = any
    sheet: str | None
    column: str               # column name as written in the README
    description: str


@dataclass
class ReadmeInfo:
    entries: list[DictEntry] = field(default_factory=list)
    conversions: list[tuple[str, str, float, str]] = field(default_factory=list)   # product, big unit, factor, base unit
    data_end: date | None = None
    data_start: date | None = None
    conventions: list[str] = field(default_factory=list)


_SECTION = re.compile(r"^#{2,4}\s*(?:Cột|Cot|Columns?)\s+(?:trong|của|cua|in)\s+(.+)$", re.I)
_TICK = re.compile(r"`([^`]+)`")
_ENTRY = re.compile(r"^\s*[-*]\s+\*\*(.+?)\*\*\s*[:：]\s*(.+)$")
_INLINE = re.compile(r"\*\*([^*]+)\*\*\s+(?:là|la)\s+([^;.]+)")
_DOTTED = re.compile(r"`([A-Za-z0-9_]+)\.([A-Za-z0-9_]+)`\s+(?:là|la)\s+([^;`]+)")
_CONV = re.compile(r"^\s*[-*]\s+(.+?)\s*:\s*1\s+([^\s=]+(?:\s[^\s=]+)?)\s*=\s*([\d.,]+)\s+(.+?)\s*\.?$")
_DATE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")


def _d(m) -> date | None:
    try:
        return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    except ValueError:
        return None


def parse_readme(text: str) -> ReadmeInfo:
    info = ReadmeInfo()
    cur_file = cur_sheet = None
    for raw in (text or "").splitlines():
        line = raw.rstrip()
        m = _SECTION.match(line)
        if m:
            ticks = _TICK.findall(m.group(1))
            cur_file = fold(re.sub(r"\.[a-z0-9]+$", "", ticks[0])) if ticks else None
            cur_sheet = fold(ticks[1]) if len(ticks) > 1 else None
            continue
        if re.match(r"^#{1,4}\s", line):
            cur_file = cur_sheet = None
            if not re.search(r"quy uoc|ghi chu|luu y", fold(line)):
                pass
            continue
        m = _ENTRY.match(line)
        if m and cur_file is not None:
            info.entries.append(DictEntry(cur_file, cur_sheet, m.group(1).strip(), m.group(2).strip()))
            continue
        for t, c, d in _DOTTED.findall(line):
            info.entries.append(DictEntry(fold(t), None, c, d.strip()))
        if cur_file is None:
            for c, d in _INLINE.findall(line):
                if len(c) <= 40:
                    info.entries.append(DictEntry(None, None, c.strip(), d.strip()))
        m = _CONV.match(line)
        if m:
            try:
                info.conversions.append((m.group(1).strip(), m.group(2).strip(), float(m.group(3).replace(",", ".")),
                                         m.group(4).strip()))
            except ValueError:
                pass
            continue
        f = fold(line)
        if re.search(r"du lieu (co )?(den|toi) het", f):
            ds = list(_DATE.finditer(line))
            if ds:
                info.data_end = _d(ds[-1])
        if re.search(r"du lieu (co )?tu", f):
            ds = list(_DATE.finditer(line))
            if ds:
                info.data_start = _d(ds[0])
        if line.strip().startswith(("-", "*")) and cur_file is None and len(line) < 400:
            info.conventions.append(line.strip(" -*"))
    return info


def match_role(description: str, desc_index: list[tuple[str, str]]) -> str | None:
    """Longest ontology phrase that STARTS the folded description (descriptions lead with their role)."""
    f = fold(description)
    for phrase, role in desc_index:
        if f == phrase or f.startswith(phrase + " ") or f.startswith(phrase):
            return role
    return None


def build_desc_index(ontology: dict) -> list[tuple[str, str]]:
    idx = [(fold(p), role) for role, phrases in (ontology.get("readme_desc") or {}).items() for p in phrases]
    return sorted(idx, key=lambda x: (-len(x[0]), x[1]))


def readme_roles(info: ReadmeInfo, prof: dict, ontology: dict) -> dict[tuple[str, str], str]:
    """(table, column) -> role for columns described in the README data dictionary."""
    idx = build_desc_index(ontology)
    out: dict[tuple[str, str], str] = {}
    for e in info.entries:
        role = match_role(e.description, idx)
        if role is None:
            continue
        key = fold(e.column)
        for t in prof["tables"]:
            src = fold(t.get("source", "").split("@")[0].replace("_", " "))
            tname = t["name"].replace("_", " ")
            if e.file and fold(e.file.replace("_", " ")) not in src and fold(e.file.replace("_", " ")) not in tname:
                continue
            if e.sheet and e.sheet.replace("_", " ") not in src and e.sheet.replace("_", " ") not in tname:
                continue
            for c in t.get("columns", []):
                last = fold(c["orig_name"].split("_")[-1])        # 2-level headers "Đơn_Mã phiếu"
                if fold(c["orig_name"]) == key or c["name"] == key.replace(" ", "_") or last == key:
                    out[(t["name"], c["name"])] = role
    return out
