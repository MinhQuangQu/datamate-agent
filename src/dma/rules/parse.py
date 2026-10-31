"""rulebook.md -> items (05.3 §1.1) and item diff vs the reference rulebook (§1.2)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from rapidfuzz import fuzz

from dma.textnorm import fold


@dataclass
class Item:
    item_id: str
    section: str
    text: str

    def as_json(self) -> dict:
        return asdict(self)


def item_id(text: str) -> str:
    return hashlib.sha1(fold(text).encode()).hexdigest()[:10]


_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")


def parse_rulebook(md: str) -> list[Item]:
    items: list[Item] = []
    path: list[str] = []
    para: list[str] = []
    table_header: list[str] | None = None

    def flush():
        nonlocal para
        txt = " ".join(s.strip() for s in para).strip()
        if txt:
            items.append(Item(item_id(txt), " > ".join(path), txt))
        para = []

    for raw in md.splitlines():
        line = raw.rstrip()
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            table_header = None
            lvl = len(m.group(1))
            path[:] = path[:lvl - 1] + [m.group(2).strip()]
            continue
        if line.strip().startswith("|"):
            flush()
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            if table_header is None:
                table_header = cells
                continue
            if len(cells) == 2 and all(cells):     # "| Luật | Mặc định |" tables: "<rule>: <meaning>"
                txt = f"{cells[0]}: {cells[1]}"
            else:
                txt = " | ".join(f"{h}: {c}" if h else c for h, c in zip(table_header, cells) if c)
            if txt:
                items.append(Item(item_id(txt), " > ".join(path), txt))
            continue
        table_header = None
        if not line.strip():
            flush()
            continue
        if _BULLET.match(line):
            flush()
            para = [_BULLET.sub("", line)]
            continue
        if para and line.startswith(("  ", "\t")):
            para.append(line)
            continue
        para.append(line)
    flush()
    return items


def diff_items(cur: list[Item], ref: list[Item], modified_threshold: int = 80) -> dict[str, list]:
    ref_by_fold = {fold(i.text): i for i in ref}
    out: dict[str, list] = {"same": [], "modified": [], "new": [], "removed": []}
    used_ref: set[str] = set()
    for it in cur:
        f = fold(it.text)
        if f in ref_by_fold:
            out["same"].append(it)
            used_ref.add(ref_by_fold[f].item_id)
            continue
        best, score = None, 0
        for r in ref:
            if r.item_id in used_ref:
                continue
            rf = fold(r.text)
            # token_set alone scores 100 whenever one text's tokens are a subset of the other's
            s = 0.5 * fuzz.token_set_ratio(f, rf) + 0.5 * fuzz.token_sort_ratio(f, rf)
            if s > score or (s == score and best is not None and r.item_id < best.item_id):
                best, score = r, s
        if best is not None and score >= modified_threshold:
            out["modified"].append((it, best))
            used_ref.add(best.item_id)
        else:
            out["new"].append((it, None))
    out["removed"] = [r for r in ref if r.item_id not in used_ref]
    return out
