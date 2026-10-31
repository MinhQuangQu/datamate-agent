"""Rebuild the knowledge pack of some businesses deterministically (no LLM) into an existing work dir
(dev tool: fast iteration on ingestion / semantic layer without rebuilding a whole split).

    python scripts/build_one.py --split data/raw/train --work /tmp/dma_build_train biz_121 biz_107
(--work must already hold `split/` from a `dma.run --no-llm` build of the same split)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dma.context import SplitContext  # noqa: E402
from dma.semantic.bkp import finalize_business, prepare_business  # noqa: E402
from dma.semantic.bsm import BSM  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("businesses", nargs="+")
    a = ap.parse_args(argv)
    split = SplitContext.load(os.path.join(a.work, "split"))
    for b in a.businesses:
        t0 = time.monotonic()
        prep = prepare_business(b, os.path.join(a.split, "data", b), os.path.join(a.work, "biz", b))
        if not prep.get("ok"):
            print(b, "prepare failed", prep.get("warnings", [])[:3])
            continue
        bkp = finalize_business(prep, BSM.model_validate(prep["proposal"]), split)
        print(f"{b}: {time.monotonic() - t0:.1f}s quality={bkp.quality} views={len(bkp.views)} warnings={bkp.warnings[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
