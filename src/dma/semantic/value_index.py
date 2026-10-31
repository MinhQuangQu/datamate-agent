"""Value index (05.2 §7): every name/code spelling in the data, folded for matching.
Outputs always use `name` or one of `variants` — strings that exist in the data."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from dma.ingest.loader import q
from dma.textnorm import fold, scorer_norm

KIND_SOURCES = [   # kind, SQL returning (code, name, freq) rows
    ("product", "SELECT p.product_id, p.product_name, COALESCE(f.n, 0) FROM sem.products p LEFT JOIN "
                "(SELECT product_id, COUNT(*) n FROM sem.sales_lines GROUP BY 1) f USING (product_id)", "sem.products"),
    ("customer", "SELECT c.customer_id, c.customer_name, COALESCE(f.n, 0) FROM sem.customers c LEFT JOIN "
                 "(SELECT customer_id, COUNT(*) n FROM sem.sales_lines GROUP BY 1) f USING (customer_id)", "sem.customers"),
    ("category", "SELECT NULL, category, (SELECT COUNT(*) FROM sem.sales_lines s WHERE s.category = c.category) "
                 "FROM sem.categories c", "sem.categories"),
    ("branch", "SELECT code, name, 0 FROM sem.branches", "sem.branches"),
    ("staff", "SELECT code, name, 0 FROM sem.staff", "sem.staff"),
    ("channel", "SELECT NULL, name, 0 FROM sem.channels", "sem.channels"),
    ("payment_method", "SELECT NULL, name, 0 FROM sem.payment_methods", "sem.payment_methods"),
]
VARIANT_SOURCES = {
    "product": "SELECT DISTINCT product_id, product_name FROM sem.sales_lines WHERE product_name IS NOT NULL",
    "customer": "SELECT DISTINCT customer_id, customer_name FROM sem.sales_lines WHERE customer_name IS NOT NULL",
}
COMMON_TOKENS = {"nuoc", "banh", "tra", "sua", "ca", "phe", "keo", "bia", "com", "mi", "thit", "rau", "chai", "lon",
                 "hop", "goi", "bao", "kg", "g", "ml", "l", "loai", "size", "lon", "nho", "vua", "chi", "anh", "co", "chu",
                 "ba", "ong", "em", "bac"}


@dataclass
class ValueEntry:
    kind: str
    code: str | None
    name: str
    variants: list[str] = field(default_factory=list)
    fold: str = ""
    tokens: list[str] = field(default_factory=list)
    freq: int = 0


class ValueIndex:
    def __init__(self, entries: list[ValueEntry]):
        self.entries = entries
        self.by_token: dict[str, list[int]] = defaultdict(list)
        self.exact: dict[str, list[int]] = defaultdict(list)       # scorer_norm(code|name|variant) -> entries
        self.by_fold: dict[str, list[int]] = defaultdict(list)     # fold(code|name|variant) -> entries
        df: dict[str, int] = defaultdict(int)
        for i, e in enumerate(entries):
            keys = {e.name, *(e.variants or [])} | ({e.code} if e.code else set())
            for k in keys:
                self.exact[scorer_norm(k)].append(i)
                self.by_fold[fold(k)].append(i)
            toks = set(e.tokens)
            for t in toks:
                self.by_token[t].append(i)
                df[t] += 1
        n = max(1, len(entries))
        self.idf = {t: math.log(1 + n / c) for t, c in df.items()}

    # --------------------------------------------------------------- lookups
    def lookup_exact(self, s: str) -> ValueEntry | None:
        ids = self.exact.get(scorer_norm(s)) or []
        if not ids:
            return None
        return sorted((self.entries[i] for i in ids), key=lambda e: (_KIND_PRI.get(e.kind, 9), -e.freq, e.code or ""))[0]

    def is_data_string(self, s: str) -> bool:
        return scorer_norm(s) in self.exact

    def lookup_fold(self, s: str) -> list[ValueEntry]:
        return [self.entries[i] for i in self.by_fold.get(fold(s), [])]

    def code_for(self, kind: str, value: str) -> str | None:
        """Map a name/code typed by the LLM to the entity CODE (or canonical name if no code)."""
        if value is None:
            return None
        for i in self.exact.get(scorer_norm(value), []) + self.by_fold.get(fold(value), []):
            e = self.entries[i]
            if e.kind == kind:
                return e.code or e.name
        return None

    def candidates(self, tokens: set[str]) -> set[int]:
        out: set[int] = set()
        for t in tokens:
            out.update(self.by_token.get(t, []))
        return out

    def to_json(self) -> list[dict]:
        return [asdict(e) for e in self.entries]

    @classmethod
    def from_json(cls, rows: list[dict]) -> "ValueIndex":
        return cls([ValueEntry(**r) for r in rows])

    def kinds(self) -> dict[str, int]:
        c: dict[str, int] = defaultdict(int)
        for e in self.entries:
            c[e.kind] += 1
        return dict(c)


_KIND_PRI = {"product": 0, "customer": 1, "category": 2, "branch": 3, "staff": 4, "supplier": 5, "channel": 6,
             "payment_method": 7}


def tokens_of(s: str) -> list[str]:
    return [t for t in fold(s).split() if t]


def build_value_index(con, views: list[str], prof: dict | None = None, max_generic: int = 5000,
                      skip_cols: set[tuple[str, str]] | None = None) -> ValueIndex:
    entries: list[ValueEntry] = []
    seen: set[tuple[str, str]] = set()
    have = set(views)
    for kind, sql, view in KIND_SOURCES:
        if view not in have:
            continue
        try:
            rows = con.execute(sql).fetchall()
        except Exception:  # noqa: BLE001
            continue
        variants: dict[str, set[str]] = defaultdict(set)
        if kind in VARIANT_SOURCES and "sem.sales_lines" in have:
            for code, nm in con.execute(VARIANT_SOURCES[kind]).fetchall():
                if code is not None and nm:
                    variants[str(code)].add(str(nm))
        for code, name, freq in rows:
            if name is None and code is None:
                continue
            name = str(name if name is not None else code)
            code = None if code is None else str(code)
            key = (kind, code or scorer_norm(name))
            if key in seen:
                continue
            seen.add(key)
            vs = sorted({v for v in variants.get(code or "", set()) if v != name})
            entries.append(ValueEntry(kind, code, name, vs, fold(name), tokens_of(name + " " + " ".join(vs)) +
                                      ([fold(code)] if code else []), int(freq or 0)))
    # generic string columns of small tables (unknown business types)
    if prof:
        skip = skip_cols or set()
        known_norm = {scorer_norm(e.name) for e in entries}
        for t in prof["tables"]:
            for c in t.get("columns", []):
                if (t["name"], c["name"]) in skip:
                    continue
                if c["kind"] != "string" or not c.get("distinct") or c["distinct"] > max_generic:
                    continue
                if (c.get("avg_len") or 0) > 80 or c["distinct"] < 2:
                    continue
                kind = f"custom:{t['name']}.{c['name']}"
                rows = con.execute(f"SELECT {q(c['name'])}, COUNT(*) FROM typed.{q(t['name'])} WHERE {q(c['name'])} "
                                   f"IS NOT NULL GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT {max_generic}").fetchall()
                for v, n in rows:
                    v = str(v)
                    k = ("custom", scorer_norm(v))
                    if k in seen or k[1] in known_norm:
                        continue
                    seen.add(k)
                    entries.append(ValueEntry(kind, None, v, [], fold(v), tokens_of(v), int(n)))
    return ValueIndex(entries)


def is_common(token: str) -> bool:
    return token in COMMON_TOKENS or bool(re.fullmatch(r"\d{1,2}", token))
