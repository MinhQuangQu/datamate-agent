"""Business Knowledge Pack (04 §5): build in three phases so CPU work overlaps model load.

  1. prepare_business()  — CPU, runs in a spawn process pool: ingest -> profile -> heuristic BSM -> Data Card
  2. refine (async)      — LLM: BSM samples + vote, status-map repair (skipped when the LLM is off)
  3. finalize_business() — CPU (thread): views per RuleSpec -> V1..V9 -> catalog -> value index -> Business Card
Never raises: a business that fails to build still gets a minimal pack (typed.* only) so
its questions can run through the Code Solver (FR-KN-09).
"""
from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from dma.context import SplitContext
from dma.ingest.loader import IngestResult, ingest_business, ingest_result_from_json
from dma.profile.datacard import render_data_card
from dma.profile.profiler import profile_business
from dma.semantic.bsm import BSM
from dma.semantic.card import render_business_card
from dma.semantic.catalog import build_catalog, catalog_from_json, catalog_to_json
from dma.semantic.heuristics import propose_bsm
from dma.semantic.validate import sanitize_roles, validate
from dma.semantic.value_index import ValueIndex, build_value_index
from dma.semantic.views import ViewBuild, build_views
from dma.solvers.plan.compiler import Catalog


@dataclass
class BusinessKnowledgePack:
    business_id: str
    dir: str
    db_path: str
    bsm: BSM
    catalog: Catalog
    vindex: ValueIndex
    card: str
    datacard: str
    profile: dict
    readme: str = ""
    views: list[str] = field(default_factory=list)
    view_cols: dict[str, list[str]] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    quality: float = 0.0
    build_s: float = 0.0
    ok: bool = True
    _con: Any = None

    def con(self):
        """Shared read-only connection; use .cursor() per thread."""
        if self._con is None:
            import duckdb
            self._con = duckdb.connect(self.db_path, read_only=True)
            self._con.execute("SET threads = 2")
        return self._con

    def cursor(self):
        return self.con().cursor()

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    def attach_lexicon(self, rulebook_text: str = "") -> None:
        from dma.textnorm import fold
        self.vindex._extra_words = set(fold(self.readme).split()) | set(fold(rulebook_text).split())

    @property
    def coverage(self) -> tuple[str, str] | None:
        cov = self.bsm.date_coverage
        if cov and self.bsm.data_end:     # the README states where the data ends
            start = self.bsm.data_start or min(a for a, _ in cov.values())
            return start, self.bsm.data_end
        if not cov:
            r = self.profile.get("date_range")
            return tuple(r) if r else None
        return min(a for a, _ in cov.values()), max(b for _, b in cov.values())

    def save(self) -> None:
        d = self.dir
        _dump(os.path.join(d, "bsm.json"), self.bsm.model_dump())
        _dump(os.path.join(d, "catalog.json"), catalog_to_json(self.catalog))
        _dump(os.path.join(d, "value_index.json"), self.vindex.to_json())
        _dump(os.path.join(d, "bkp.json"), {"business_id": self.business_id, "db_path": self.db_path, "views": self.views,
                                            "view_cols": self.view_cols, "notes": self.notes, "warnings": self.warnings,
                                            "quality": self.quality, "build_s": self.build_s, "ok": self.ok})
        with open(os.path.join(d, "business_card.md"), "w", encoding="utf-8") as f:
            f.write(self.card)


def _dump(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=str, sort_keys=True)
    os.replace(tmp, path)


def load_bkp(d: str) -> BusinessKnowledgePack:
    with open(os.path.join(d, "bkp.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(d, "bsm.json"), encoding="utf-8") as f:
        bsm = BSM.model_validate(json.load(f))
    with open(os.path.join(d, "catalog.json"), encoding="utf-8") as f:
        cat = catalog_from_json(json.load(f))
    with open(os.path.join(d, "value_index.json"), encoding="utf-8") as f:
        vi = ValueIndex.from_json(json.load(f))
    with open(os.path.join(d, "profile.json"), encoding="utf-8") as f:
        prof = json.load(f)
    card = open(os.path.join(d, "business_card.md"), encoding="utf-8").read()
    datacard = open(os.path.join(d, "datacard.md"), encoding="utf-8").read()
    readme = open(os.path.join(d, "readme.md"), encoding="utf-8").read() if os.path.exists(os.path.join(d, "readme.md")) else ""
    return BusinessKnowledgePack(meta["business_id"], d, meta["db_path"], bsm, cat, vi, card, datacard, prof, readme,
                                 meta["views"], meta["view_cols"], meta["notes"], meta["warnings"], meta["quality"],
                                 meta["build_s"], meta["ok"])


# ------------------------------------------------------------------ phase 1
def prepare_business(business_id: str, biz_dir: str, work_dir: str, max_card_tokens: int = 2500) -> dict:
    """Top-level (picklable) CPU phase for the spawn pool. Returns a JSON-able dict."""
    import duckdb
    t0 = time.monotonic()
    out: dict[str, Any] = {"business_id": business_id, "dir": work_dir, "ok": True, "warnings": []}
    try:
        ing = ingest_business(biz_dir, work_dir)
        out["ingest"] = ing.to_json()
        out["warnings"] += ing.warnings
        con = duckdb.connect(ing.db_path, read_only=True)
        try:
            prof = profile_business(ing, con)
            proposal = propose_bsm(prof, con, ing.readme)
        finally:
            con.close()
        readme = ing.readme + ("\n\n" + "\n\n".join(ing.docs) if ing.docs else "")
        datacard = render_data_card(prof, readme, max_tokens=max_card_tokens)
        _dump(os.path.join(work_dir, "profile.json"), prof)
        _dump(os.path.join(work_dir, "proposal.json"), proposal.model_dump())
        with open(os.path.join(work_dir, "datacard.md"), "w", encoding="utf-8") as f:
            f.write(datacard)
        with open(os.path.join(work_dir, "readme.md"), "w", encoding="utf-8") as f:
            f.write(readme)
        out.update(profile=prof, proposal=proposal.model_dump(), datacard=datacard, readme=readme)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["warnings"].append(f"prepare failed: {type(e).__name__}: {e}")
        out["traceback"] = traceback.format_exc()[-3000:]
    out["prepare_s"] = round(time.monotonic() - t0, 3)
    return out


# ------------------------------------------------------------------ phase 3
def _has_values(con, view: str, col: str) -> bool:
    try:
        return bool(con.execute(f"SELECT COUNT({col}) FROM {view}").fetchone()[0])
    except Exception:  # noqa: BLE001
        return False


def finalize_business(prep: dict, bsm: BSM, split: SplitContext, extra_warnings: list[str] | None = None) -> BusinessKnowledgePack:
    import duckdb
    t0 = time.monotonic()
    d = prep["dir"]
    ing: IngestResult = ingest_result_from_json(dict(prep["ingest"]))
    prof = prep["profile"]
    warns = list(prep.get("warnings", [])) + list(extra_warnings or [])
    warns += sanitize_roles(bsm, prof)
    vb = ViewBuild()
    quality, checks = 0.0, {}
    con = duckdb.connect(ing.db_path)
    try:
        try:
            vb = build_views(con, bsm, split.rulespec, prof)
        except Exception as e:  # noqa: BLE001 - fall back to generic layer only
            warns.append(f"semantic views failed: {type(e).__name__}: {e}")
            for v in ("sales_lines_valid", "order_flags", "orders", "orders_all", "sales_lines", "_lines_base"):
                try:
                    con.execute(f"DROP VIEW IF EXISTS sem.{v}")
                except Exception:  # noqa: BLE001
                    pass
            vb = ViewBuild()
        vwarn, quality, checks = validate(con, bsm, vb.views, vb.columns)
        warns += vwarn + vb.warnings
        generic = {f"typed.{t['name']}": next((c["name"] for c in t["columns"] if c["sql_type"] in ("DATE", "TIMESTAMP")), "")
                   for t in prof["tables"]}
        cat = build_catalog(bsm, split.rulespec, vb, split.weekday_names, generic)
        # a gap is "the data cannot answer this": drop the ones the built semantic layer does answer
        covered = {"receivables": "debt_outstanding" in cat.metrics, "inventory": "stock_on_hand" in cat.metrics,
                   "cost": "cogs" in cat.metrics,
                   "expiry": "sem.purchase_lots" in vb.views and _has_values(con, "sem.purchase_lots", "expiry_date")}
        bsm.gaps = [g for g in bsm.gaps if not covered.get(g, False)]
        skip = {(r.table, r.column) for r in bsm.roles}
        vindex = build_value_index(con, vb.views, prof, skip_cols=skip)
        bsm.warnings = list(dict.fromkeys(list(bsm.warnings) + warns))[:40]
        bsm.quality = quality
        bsm.interpretation = dict(vb.notes)
        card = render_business_card(con, bsm, cat, vb.views, vb.columns, prof, prep.get("readme", ""), vb.notes)
        con.execute("CHECKPOINT")
    finally:
        con.close()
    with open(os.path.join(d, "views.sql"), "w", encoding="utf-8") as f:
        f.write(";\n\n".join(vb.sql))
    bkp = BusinessKnowledgePack(prep["business_id"], d, ing.db_path, bsm, cat, vindex, card, prep["datacard"], prof,
                                prep.get("readme", ""), vb.views, vb.columns, vb.notes, warns, quality,
                                round(prep.get("prepare_s", 0) + time.monotonic() - t0, 3), True)
    bkp.save()
    _dump(os.path.join(d, "validation.json"), checks)
    return bkp


def minimal_bkp(prep: dict, split: SplitContext) -> BusinessKnowledgePack | None:
    """Business whose preparation failed: empty semantic layer (generic Code Solver only)."""
    try:
        bsm = BSM(warnings=prep.get("warnings", [])[:20], gaps=[])
        prep = dict(prep)
        if "ingest" not in prep:
            return None
        prep.setdefault("profile", {"tables": [], "fks": [], "files": []})
        prep.setdefault("datacard", "# DATA CARD\n(unavailable)")
        return finalize_business(prep, bsm, split)
    except Exception:  # noqa: BLE001
        return None
