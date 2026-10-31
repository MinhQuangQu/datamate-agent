# DataMate Agent — developer shortcuts (10-setup-guide.md §7)
PY ?= .venv/bin/python
TRAIN ?= data/raw/train
ENDPOINTS ?= http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1
MODEL ?= /models/qwen3.5-9b

.PHONY: test lint synth serve dev-fake dev cv calibrate exemplars docker

test:            ## unit + integration + e2e (fake LLM)
	$(PY) -m pytest -q

lint:            ## Rule 4 guard
	$(PY) scripts/check_no_id_logic.py src/ synth/ scripts/

synth:           ## small synthetic split with answers
	PYTHONPATH=src:. $(PY) -m synth.generator --out data/synth/v1 --businesses 6 --questions 40

serve:           ## one vLLM replica per GPU on :8001, :8002
	MODEL=$(MODEL) bash scripts/serve_llm.sh

dev-fake: synth  ## whole pipeline with the FakeBackend test double
	$(PY) -m dma.run --config configs/dev.yaml --data data/synth/v1 --result runs/dev_fake --fake-llm
	$(PY) -m dma.eval.scorer --pred runs/dev_fake/submission.jsonl --gold data/synth/v1/answers.jsonl

dev:             ## real model on the train split (servers from `make serve`)
	$(PY) -m dma.run --config configs/dev.yaml --data $(TRAIN) --result runs/dev --endpoints $(ENDPOINTS)

cv:              ## leave-business-out CV with the real model
	$(PY) scripts/run_cv.py --split $(TRAIN) --folds 5 --endpoints $(ENDPOINTS) --out runs/lobo

exemplars:
	$(PY) scripts/mine_exemplars.py --runs runs/lobo --split $(TRAIN) --out assets/exemplars.jsonl --keep-seed

calibrate:
	$(PY) scripts/train_calibrator.py runs/lobo/oof_features.jsonl --out assets/calibrator.json

docker:
	docker build -t datamate-agent:latest .
