"""Draft assets/train_rulespec.json from the TRAIN rulebook (05.3 §6), deterministically.

    python scripts/build_reference_rulespec.py data/raw/train/rulebook.md
Then: review every field + provenance by hand, fix what the extractor missed, run the rule
tests, and copy the rulebook to assets/train_rulebook.md (the diff reference).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.rules.extract import bootstrap_rulespec  # noqa: E402
from dma.rules.parse import parse_rulebook  # noqa: E402
from dma.rules.rulespec import RuleSpec, render_digest  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rulebook")
    ap.add_argument("--out", default="assets/train_rulespec.json")
    ap.add_argument("--copy-to", default="assets/train_rulebook.md")
    a = ap.parse_args(argv)
    md = open(a.rulebook, encoding="utf-8-sig").read()
    spec = bootstrap_rulespec(md)
    default = RuleSpec().model_dump()
    got = spec.model_dump()
    changed = {k: got[k] for k in default if k not in ("provenance", "free_text_rules", "vocabulary") and got[k] != default[k]}
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(spec.model_dump_json(indent=1))
    if a.copy_to and os.path.abspath(a.copy_to) != os.path.abspath(a.rulebook):
        shutil.copy(a.rulebook, a.copy_to)
    items = parse_rulebook(md)
    mapped = {i for ids in spec.provenance.values() for i in ids}
    print(f"[rulespec] {len(items)} items, {len(mapped)} mapped to fields, {len(spec.free_text_rules)} verbatim, "
          f"{len(spec.vocabulary)} vocabulary")
    print("[rulespec] fields different from portal defaults:", json.dumps(changed, ensure_ascii=False))
    print("[rulespec] UNMAPPED items (check by hand):")
    for it in items:
        if it.item_id not in mapped:
            print(f"  - [{it.item_id}] {it.section}: {it.text}")
    print("\n" + render_digest(spec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
