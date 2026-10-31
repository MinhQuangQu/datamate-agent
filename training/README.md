# Training code shipped in the image (/app/training)

Nothing here runs at inference time. Everything is reproducible from the train split.

| Step | Command | Output |
|---|---|---|
| Reference RuleSpec from the train rulebook (then hand-verify) | `python scripts/build_reference_rulespec.py data/raw/train/rulebook.md` | `assets/train_rulespec.json`, `assets/train_rulebook.md` |
| Leave-business-out CV with the served model | `python scripts/run_cv.py --split data/raw/train --folds 5 --endpoints http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1 --out runs/lobo` | `runs/lobo/{summary.md,items.csv,oof_features.jsonl}` |
| Exemplar mining (verified plans / code) | `python scripts/mine_exemplars.py --runs runs/lobo --split data/raw/train --out assets/exemplars.jsonl --keep-seed` | `assets/exemplars.jsonl` |
| Calibrator + thresholds | `python scripts/train_calibrator.py runs/lobo/oof_features.jsonl --out assets/calibrator.json` | `assets/calibrator.json` |
| Synthetic new-business-type data | `python -m synth.generator --out data/synth/newtypes --businesses 20 --questions 100 --kinds tiem_hoa` | split with answers |

Optional LoRA fine-tuning (06 §7) is not part of this release.
