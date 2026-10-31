import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


@pytest.fixture(scope="session")
def synth_split(tmp_path_factory):
    """Small synthetic split (3 businesses: xlsx / sqlite / flat csv) with answers."""
    from synth.generator import main as gen
    out = tmp_path_factory.mktemp("synth")
    gen(["--out", str(out), "--businesses", "3", "--questions", "14", "--seed", "5", "--months", "14"])
    return str(out)


@pytest.fixture(scope="session")
def built(synth_split, tmp_path_factory):
    """BKPs + saved split context for the synthetic split (no LLM)."""
    from dma.context import SplitContext, load_calendar
    from dma.output.formatter import FormatConfig
    from dma.rules.extract import load_reference
    from dma.semantic.bkp import finalize_business, prepare_business
    from dma.semantic.bsm import BSM
    from dma.semantic.catalog import load_weekday_names
    work = tmp_path_factory.mktemp("work")
    _, ref = load_reference(os.path.join(ROOT, "assets/train_rulebook.md"), os.path.join(ROOT, "assets/train_rulespec.json"))
    cal, _ = load_calendar(os.path.join(synth_split, "calendar.csv"))
    split = SplitContext.make(ref, cal, load_weekday_names(os.path.join(ROOT, "assets/weekday_names.json")), FormatConfig(),
                              open(os.path.join(synth_split, "rulebook.md")).read(), calendar_path=os.path.join(synth_split, "calendar.csv"))
    split.save(str(work / "split"))
    bkps = {}
    for b in sorted(os.listdir(os.path.join(synth_split, "data"))):
        prep = prepare_business(b, os.path.join(synth_split, "data", b), str(work / "biz" / b))
        assert prep["ok"], prep.get("traceback")
        bkps[b] = finalize_business(prep, BSM.model_validate(prep["proposal"]), split)
    return {"split": split, "bkps": bkps, "work": str(work), "data": synth_split}
