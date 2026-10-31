"""Split RuleSpec builder (05.3 §1–2): parse -> diff vs reference -> reuse same items ->
re-extract only changed/new items (deterministic extractor + LLM samples, field-wise vote)
-> sanity checks -> RuleSpec + report (goes into run_manifest.rulebook_diff_vs_train)."""
from __future__ import annotations

import asyncio
import json
import os
import re
from collections import Counter
from typing import Any

from dma.llm import prompts as P
from dma.rules.heuristic import extract_patch, extract_vocab
from dma.rules.parse import Item, diff_items, parse_rulebook
from dma.rules.rulespec import SCALAR_FIELDS, RuleSpec, VocabEntry, field_schema
from dma.textnorm import fold

HEURISTIC_WEIGHT = 2.0


def load_reference(rulebook_path: str | None, rulespec_path: str | None) -> tuple[list[Item], RuleSpec]:
    items: list[Item] = []
    if rulebook_path and os.path.exists(rulebook_path):
        with open(rulebook_path, encoding="utf-8") as f:
            items = parse_rulebook(f.read())
    spec = RuleSpec()
    if rulespec_path and os.path.exists(rulespec_path):
        with open(rulespec_path, encoding="utf-8") as f:
            spec = RuleSpec.model_validate(json.load(f))
    return items, spec


def bootstrap_rulespec(md: str, base: RuleSpec | None = None) -> RuleSpec:
    """Deterministic RuleSpec from a whole rulebook (used to draft train_rulespec.json)."""
    spec = base or RuleSpec()
    items = parse_rulebook(md)
    patch, prov, free, vocab = {}, {}, [], []
    for it in items:
        p, touched = extract_patch(it.text)
        for k, v in p.items():
            patch[k] = v
            prov.setdefault(k, []).append(it.item_id)
        if not p:
            ve = extract_vocab(it.text) if re.search(r"tu ngu|thuat ngu|tu vung|vocabulary", fold(it.section)) else None
            if ve:
                vocab.append(VocabEntry(**ve))
            elif not touched:
                free.append(it.text)
    spec = spec.with_patch(patch)
    return spec.model_copy(update={"provenance": prov, "free_text_rules": free, "vocabulary": vocab})


def sanity(spec: RuleSpec) -> list[str]:
    errs = []
    t = spec.time
    if t.tet_start_lunar[0] not in (11, 12, 1) or not 1 <= t.tet_start_lunar[1] <= 30:
        errs.append("time.tet_start_lunar")
    if t.tet_end_lunar[0] not in (1, 2) or not 1 <= t.tet_end_lunar[1] <= 30:
        errs.append("time.tet_end_lunar")
    if any(not 1 <= d <= 7 for d in t.weekend_isodays) or not t.weekend_isodays:
        errs.append("time.weekend_isodays")
    if not 0 <= spec.numbers.decimals <= 4:
        errs.append("numbers.decimals")
    if not -6 <= spec.numbers.money_decimals <= 4:
        errs.append("numbers.money_decimals")
    if not spec.valid_order.canceled_statuses:
        errs.append("valid_order.canceled_statuses")
    return errs


def _key(v: Any) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False)


async def build_split_rulespec(rulebook_md: str | None, ref_items: list[Item], ref_spec: RuleSpec,
                               llm=None, n_samples: int = 5, min_share: float = 0.8,
                               seed_base: int = 0) -> tuple[RuleSpec, dict]:
    report: dict[str, Any] = {"source": "reference", "same": 0, "modified": [], "new": [], "removed": [],
                              "changed_sections": [], "patch": {}, "uncertain": [], "llm_used": False}
    if not rulebook_md:
        report["source"] = "reference(no rulebook in split)"
        spec = ref_spec.model_copy(update={"uncertain_fields": sorted(set(ref_spec.uncertain_fields) | {"rulebook_missing"})})
        return spec, report
    cur = parse_rulebook(rulebook_md)
    if not ref_items:                         # no reference shipped: everything is "new" against defaults
        d = {"same": [], "modified": [], "new": [(it, None) for it in cur], "removed": []}
    else:
        d = diff_items(cur, ref_items)
    report["same"] = len(d["same"])
    report["modified"] = [it.item_id for it, _ in d["modified"]]
    report["new"] = [it.item_id for it, _ in d["new"]]
    report["removed"] = [r.item_id for r in d["removed"]]
    report["changed_sections"] = sorted({it.section for it, _ in d["modified"] + d["new"]})

    changed: list[tuple[Item, Item | None]] = d["modified"] + d["new"]
    spec = ref_spec
    if not changed and not d["removed"]:
        return spec, report

    # field -> ref item ids (provenance) to know which fields a modified item touches
    prov_by_item: dict[str, list[str]] = {}
    for fld, ids in ref_spec.provenance.items():
        for i in ids:
            prov_by_item.setdefault(i, []).append(fld)

    votes: dict[str, list[tuple[str, float, Any]]] = {}       # field -> [(key, weight, value)]
    touched_by: dict[str, list[str]] = {}
    unmapped: set[str] = set()
    new_vocab: list[VocabEntry] = []
    for it, ref in changed:
        patch, touched = extract_patch(it.text)
        fields = set(touched) | set(prov_by_item.get(ref.item_id, []) if ref else [])
        for f in fields:
            touched_by.setdefault(f, []).append(it.item_id)
        for k, v in patch.items():
            votes.setdefault(k, []).append((_key(v), HEURISTIC_WEIGHT, v))
        if not fields:
            ve = extract_vocab(it.text)
            if ve:
                new_vocab.append(VocabEntry(**ve))
            unmapped.add(it.item_id)

    n_llm = 0
    if llm is not None and getattr(llm, "enabled", False) and n_samples > 0:
        fields = sorted(set(touched_by) | set(SCALAR_FIELDS if any(r is None for _, r in changed) else []))
        fields_json = json.dumps({f: {"schema": field_schema(f), "current": ref_spec.get(f)} for f in fields if f in SCALAR_FIELDS},
                                 ensure_ascii=False, default=list)
        items_json = json.dumps([{"item_id": it.item_id, "section": it.section, "text": it.text} for it, _ in changed],
                                ensure_ascii=False)
        msgs = [{"role": "system", "content": P.RULE_SYSTEM},
                {"role": "user", "content": P.RULE_USER.format(items_json=items_json, fields_json=fields_json)}]
        report["llm_used"] = True
        results = await asyncio.gather(*(llm.json("rule", msgs, schema=P.RULE_SCHEMA, seed=seed_base + 7919 * (i + 1))
                                         for i in range(n_samples)))
        for obj, err, _ in results:
            if obj is None or not isinstance(obj.get("patch"), dict):
                continue
            n_llm += 1
            for k, v in obj["patch"].items():
                if k in SCALAR_FIELDS:
                    votes.setdefault(k, []).append((_key(v), 1.0, v))
                    touched_by.setdefault(k, [])
            for iid in obj.get("unmapped", []) or []:
                unmapped.add(str(iid))
    total_w = HEURISTIC_WEIGHT + n_llm

    patch: dict[str, Any] = {}
    uncertain: set[str] = set(ref_spec.uncertain_fields)
    for fld, vs in sorted(votes.items()):
        c: Counter = Counter()
        vals = {}
        for k, w, v in vs:
            c[k] += w
            vals[k] = v
        best, w = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        denom = total_w if n_llm else sum(c.values())
        if w / max(denom, 1e-9) >= min_share or (not n_llm and w >= HEURISTIC_WEIGHT):
            patch[fld] = vals[best]
            uncertain.discard(fld)
        else:
            uncertain.add(fld)
    for fld in touched_by:
        if fld not in patch and fld in SCALAR_FIELDS:
            # an item about this field changed but no confident value: keep reference value, flag it
            uncertain.add(fld)
    for r in d["removed"]:
        for fld in prov_by_item.get(r.item_id, []):
            if fld not in patch:
                uncertain.add(fld)
    try:
        spec = ref_spec.with_patch(patch)
    except Exception as e:  # noqa: BLE001
        report["patch_error"] = repr(e)
        spec = ref_spec
    bad = sanity(spec)
    if bad:
        for b in bad:
            patch.pop(b, None)
        spec = ref_spec.with_patch(patch)
        uncertain |= set(bad)
    prov = {k: list(v) for k, v in ref_spec.provenance.items()}
    for fld, ids in touched_by.items():
        if fld in patch:
            prov[fld] = sorted(set(ids))
    cur_by_id = {it.item_id: it for it in cur}
    removed_texts = {r.text for r in d["removed"]}
    free = [x for x in ref_spec.free_text_rules if x not in removed_texts]
    for iid in sorted(unmapped):
        if iid in cur_by_id and cur_by_id[iid].text not in free:
            free.append(cur_by_id[iid].text)
    for fld in sorted(uncertain):     # uncertain fields: show the verbatim items to the solvers
        for iid in touched_by.get(fld, []):
            if iid in cur_by_id and cur_by_id[iid].text not in free:
                free.append(cur_by_id[iid].text)
    vocab = list(ref_spec.vocabulary) + [v for v in new_vocab if v.term not in {x.term for x in ref_spec.vocabulary}]
    spec = spec.model_copy(update={"provenance": prov, "free_text_rules": free, "vocabulary": vocab,
                                   "uncertain_fields": sorted(uncertain)})
    report.update(patch=patch, uncertain=sorted(uncertain), llm_samples=n_llm, source="reference+diff")
    return spec, report
