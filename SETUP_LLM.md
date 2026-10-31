# DataMate Agent — setting up the LLM on a GPU server (SSH)

This is the runbook for taking the current workspace to a GPU machine, plugging a model in, and measuring the
**real agent** (LLM + deterministic layers) on the train split. Everything except the model is already
implemented and tested on CPU (85 tests, train 30/30 and public 14/14 shops build without errors).

> **Status to keep in mind.** The pipeline has never run against a real model yet (the dev box has no GPU).
> Expect the first run to surface serving / JSON / timeout issues — section 8 lists the knobs for each.

## Quickstart (GPU server, ~5 commands)

```bash
unzip zalo_ai_project.zip && cd zalo_ai/datamate-agent
bash scripts/setup_server.sh                 # data, app venv + tests, vLLM venv, downloads Qwen/Qwen3.5-9B (≈20 GB)
source ~/vllm-env/bin/activate && MODEL=~/models/main bash scripts/serve_llm.sh   # 1 replica per GPU, waits until ready
bash scripts/run_gpu.sh smoke                # 20 questions, scored — read runs/smoke/logs/ before going further
bash scripts/run_gpu.sh train                # 3000 train questions, scored per shop (runs/train_full/score.json)
bash scripts/run_gpu.sh public               # runs/public/submission.jsonl -> upload to the portal (no local answers)
```

`MODEL_ID=… bash scripts/setup_server.sh` picks another model, `SKIP_MODEL=1` skips the download, `NGPU=1` forces one
GPU, `bash scripts/run_gpu.sh train --resume` continues an interrupted run. The sections below explain each step and
what to do when something fails (section 8).

---

## 0. What you are shipping

```
zalo_ai/
├── 01-…18-*.md, 05-detailed-design/   design docs (Vietnamese)
├── train.zip, public_test.zip          organiser data (unzipped into datamate-agent/data/raw in step 2)
├── starter-kit/, templates/, reference/
└── datamate-agent/                     the solution (Python package `dma`)
    ├── src/dma/        pipeline: ingest → semantic layer → rules → NLU → plan & code solvers → verify
    ├── configs/        dev.yaml (endpoints you start yourself), final.yaml (Docker: supervisor starts vLLM)
    ├── assets/         ontology, train RuleSpec (built from the real train rulebook), calibrator, exemplars
    ├── scripts/        serve_llm.sh, run_cv.py, probe_deterministic.py, build_one.py, validate_submission.py …
    └── tests/          pytest suite (runs without GPU, uses a fake LLM)
```

The LLM is reached **only** through an OpenAI-compatible `/v1/chat/completions` endpoint, so the agent code does not
change when you swap the model.

## 1. Server requirements

| Item | Needed |
|---|---|
| GPU | 1–2 × NVIDIA ≥ 24 GB (target hardware is 2 × RTX 5090 32 GB). One vLLM replica per GPU. |
| Driver / CUDA | whatever your vLLM build needs (`nvidia-smi` must work; Blackwell needs a CUDA 12.8+ build) |
| Python | 3.11+ for the app (dev box used 3.13). vLLM goes in its **own** venv. |
| Disk | ~25 GB model weights (BF16 9B) + ~2 GB data/work dirs |
| RAM | ≥ 16 GB (DuckDB builds 30 shops, 2 at a time by default) |
| Network | only to download the model once; the run itself is offline |

## 2. Copy and unpack

```bash
# on your laptop
scp zalo_ai_project.zip user@server:~/
# on the server
cd ~ && unzip zalo_ai_project.zip            # -> ~/zalo_ai/...
cd ~/zalo_ai/datamate-agent
mkdir -p data/raw
unzip -q ../train.zip       -d data/raw      # -> data/raw/train/{questions,answers}.jsonl, rulebook.md, data/
unzip -q ../public_test.zip -d data/raw      # -> data/raw/public_test/...
```

## 3. App environment (CPU part)

```bash
cd ~/zalo_ai/datamate-agent
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]" httpx charset-normalizer
.venv/bin/python -m pytest -q                 # expect: 85 passed
.venv/bin/python scripts/check_no_id_logic.py # expect: OK
```

If pytest fails here, fix the environment before touching the model.

## 4. Model + vLLM

### 4.1 Pick and download the model

Default per `../06-model-strategy.md`: **Qwen3.5-9B** (Apache-2.0, ≈ 9.6B params incl. vision tower). Hard rules of
the challenge: total params of every model used at inference ≤ **15B**, weights released before 2026-10-01, licence
allows use. Verify the exact HF repo id, release date and licence on the model card before committing to it.

```bash
python3 -m venv ~/vllm-env && ~/vllm-env/bin/pip install -U vllm huggingface_hub
~/vllm-env/bin/huggingface-cli download <org>/<model> --local-dir ~/models/main
# Rule 2 check (reads safetensors headers only):
cd ~/zalo_ai/datamate-agent && .venv/bin/python -m dma.utils.count_params ~/models/main --limit 15e9
```

Any vLLM version that supports the chosen model works; check flags with `~/vllm-env/bin/vllm serve --help`.

### 4.2 Start one replica per GPU

`scripts/serve_llm.sh` starts `vllm serve` on ports 8001, 8002, … with served name **`main`** (the app expects that
name, see `llm.served_name`).

```bash
cd ~/zalo_ai/datamate-agent
source ~/vllm-env/bin/activate
MODEL=~/models/main bash scripts/serve_llm.sh            # waits until every replica answers /health
# variants:
NGPU=1 MODEL=~/models/main bash scripts/serve_llm.sh                       # one GPU only
DTYPE="--quantization fp8" MODEL=~/models/main bash scripts/serve_llm.sh   # less memory, small accuracy cost
VLLM_EXTRA="" MODEL=~/models/other bash scripts/serve_llm.sh               # non-Qwen model: drop qwen3 reasoning parser
```

Logs: `logs/vllm_gpu<N>.log`. Defaults: `--max-model-len 32768 --gpu-memory-utilization 0.90 --max-num-seqs 64
--enable-prefix-caching --seed 0 --reasoning-parser qwen3`. If a GPU runs out of memory, use FP8 or lower
`--max-model-len` to 24576 inside the script.

Stop: `pkill -f "vllm serve"`.

### 4.3 Smoke-test the endpoint by hand

```bash
curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "main", "max_tokens": 200,
  "messages": [{"role": "user", "content": "Trả lời bằng JSON {\"x\": 1}"}],
  "chat_template_kwargs": {"enable_thinking": false},
  "response_format": {"type": "json_schema", "json_schema": {"name": "t", "schema": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}
}'
```

This is exactly the request shape the gateway sends (`src/dma/llm/gateway.py::_body`). It must return valid JSON in
`choices[0].message.content`; with thinking on, the reasoning must land in `message.reasoning_content` (or
`reasoning`), not inside `content`.

## 5. App configuration for the server

Copy `configs/dev.yaml` to `configs/ssh.yaml` and adjust only the `llm:` block (everything business-related comes
from the split's RuleSpec, never from config):

```yaml
llm:
  start_servers: false                       # you started vLLM yourself (step 4.2)
  endpoints: ["http://127.0.0.1:8001/v1", "http://127.0.0.1:8002/v1"]   # one entry per replica
  served_name: main
  request_timeout_s: 300
  thinking_style: qwen                       # qwen -> chat_template_kwargs.enable_thinking ; none for other models
  structured_output: response_format         # response_format | guided_json (older vLLM) | none (parse JSON from text)
```

`--endpoints a,b` on the command line overrides the list. Per-stage token budgets (`max_tokens.*`), thinking on/off
per stage (`thinking.*`) and sampling live in `configs/final.yaml` / `src/dma/config.py` defaults (06 §5 table).

## 6. First runs — smallest to largest

All commands from `~/zalo_ai/datamate-agent`, with the app venv (`.venv/bin/python`).

```bash
EP=http://127.0.0.1:8001/v1,http://127.0.0.1:8002/v1      # or just :8001 with one GPU

# 6.1 20 questions of one shop, end to end
.venv/bin/python -m dma.run --config configs/ssh.yaml --data data/raw/train --result runs/smoke \
    --work /tmp/dma_smoke --only-business biz_101 --limit 20 --endpoints $EP
less runs/smoke/logs/events.jsonl            # rulespec, bsm_refined, business_ready, per-question events
ls runs/smoke/logs/q_*.json                  # full trace per question: plan / code cells / candidates / decision
.venv/bin/python -m dma.eval.scorer --pred runs/smoke/submission.jsonl --gold data/raw/train/answers.jsonl

# 6.2 a few shops of different kinds (POS export, hand-written sheets, SQLite, pharmacy, debt ledger)
.venv/bin/python -m dma.run --config configs/ssh.yaml --data data/raw/train --result runs/dev5 \
    --work /tmp/dma_dev5 --only-business biz_101,biz_105,biz_106,biz_121,biz_123 --endpoints $EP

# 6.3 full train split (3000 questions) — this is the number that matters
.venv/bin/python -m dma.run --config configs/ssh.yaml --data data/raw/train --result runs/train_full \
    --work /tmp/dma_train --endpoints $EP
.venv/bin/python -m dma.eval.scorer --pred runs/train_full/submission.jsonl --gold data/raw/train/answers.jsonl \
    --questions data/raw/train/questions.jsonl --report runs/train_full/score.json      # per-shop breakdown
.venv/bin/python scripts/validate_submission.py runs/train_full/submission.jsonl data/raw/train/questions.jsonl
```

Scoring is +1 correct, −1 wrong, 0 abstain, so watch both accuracy and the abstain rate. Per-family breakdowns:
`scripts/run_stats.py`, run-to-run diffs: `scripts/compare_runs.py` (see `--help`).

What to read when a question is wrong: `runs/<run>/logs/<question_id>.json`. If the plan/code is right but the
value is wrong, it is a deterministic-layer bug: reproduce it without the LLM with
`scripts/probe_deterministic.py` / `scripts/build_one.py` (README, "Measuring the deterministic layers").

## 7. Day-1 work once the model runs (fills the assets the code cannot know yet)

| Step | Command | Output |
|---|---|---|
| Leave-business-out CV with the real model | `make cv ENDPOINTS=$EP` (`scripts/run_cv.py --folds 5`) | `runs/lobo/` + out-of-fold features |
| Mine exemplars from correct CV trajectories | `make exemplars` | `assets/exemplars.jsonl` (now 17 seed examples) |
| Train the abstention calibrator | `make calibrate` | `assets/calibrator.json` (now an untrained fallback table) |
| Re-run the full train split and compare | 6.3 + `scripts/compare_runs.py` | the before/after numbers |
| Run public_test (no answers) | `dma.run --data data/raw/public_test …` then `validate_submission.py` | submission for the leaderboard |

Never put public-test questions into exemplars, prompts or training data, and keep `scripts/check_no_id_logic.py`
clean: the agent must solve unseen shops (the private test) by reading their files, README and rulebook.

## 8. Troubleshooting the first real-model runs

| Symptom (events.jsonl / traces) | Likely cause | Knob |
|---|---|---|
| every question `abstain`, `LLMUnavailable` | wrong URL/port or served name | `curl …/v1/models`; `llm.endpoints`, `llm.served_name` |
| HTTP 400 on requests with a schema | vLLM rejects `response_format` with reasoning parser | `structured_output: guided_json`, else `none` |
| HTTP 400 `chat_template_kwargs` | model has no thinking switch | `thinking_style: none` |
| empty `content`, everything in reasoning | thinking budget too small | raise `max_tokens.<stage>`; the gateway already retries once without thinking |
| JSON parse errors in plans | weak structured output | `structured_output: none` (JSON then parsed from text) or a stronger model |
| timeouts / slow | too much concurrency for the GPUs | `run.max_concurrency` (dev: 8), `llm.request_timeout_s`, FP8 |
| OOM in vLLM | context + batch too large | FP8, `--max-model-len 24576`, `--max-num-seqs 32` |
| shop build failed | ingestion edge case | `grep business_build_failed runs/<run>/logs/events.jsonl`; reproduce with `scripts/build_one.py` |
| reasoning text leaks into answers | reasoning parser missing | keep `--reasoning-parser qwen3` for Qwen; the gateway also strips `<think>…</think>` |

## 8b. Trying a hosted API first (Groq, or any OpenAI-compatible provider)

For experiments only: the submission must run offline with a model ≤ 15B. `configs/groq.yaml` is ready:

```bash
export GROQ_API_KEY=...            # or keep it in datamate-agent/.env (gitignored) and: export GROQ_API_KEY=$(head -1 .env)
.venv/bin/python -m dma.run --config configs/groq.yaml --data data/samples/train_s12 --result runs/groq_s12 --work /tmp/dma_groq
.venv/bin/python -m dma.eval.scorer --pred runs/groq_s12/submission.jsonl --gold data/samples/train_s12/answers.jsonl
```

What the gateway does for hosted APIs (all in config, no code change per provider): bearer key from
`llm.api_key_env`; `llm.drop_params` removes parameters the provider rejects (Groq: `top_k`); HTTP 429 waits for the
provider's `retry-after`; `llm.rate_limit` paces requests client-side; `/v1/models` replaces `/health`.

Findings on Groq `qwen/qwen3.8-27b` (2026-10-29, free/on_demand key):
- rejects `chat_template_kwargs` and `top_k` → `thinking_style: none`, `drop_params: [top_k]`; no reasoning trace is returned;
- `response_format: json_schema` mangles nested keys of the DMQL schema (`"measure_"`, `"filters_"`) →
  `structured_output: none` (JSON parsed from text, then validated/repaired as usual);
- limits: ~7k input tokens/min and 1000 requests/day. One plan/code prompt is ~5.8–6.8k tokens even with
  `rules.prompt_rulebook: relevant`, BSM refine prompts are larger (413 → the deterministic BSM is kept), so the
  agent manages ~1 question per 3–5 minutes: enough for a smoke test, not for the 4000 train+public questions.
  A paid tier (higher TPM/RPD) is needed for full runs; raise `rate_limit`, `max_concurrency` and re-enable
  `arbitration` extras and the judge there.
- `run.question_timeout_s` (default 900) must be large when requests queue behind a rate limit.

## 8c. Single 24 GB GPU (RTX 3090 dev box, tested 2026-10-30)

```bash
# BF16 Qwen3.5-9B (19.3 GB) leaves no KV cache on 24 GB: serve FP8 weight-only (Marlin on Ampere, KV ≈ 255k tokens).
# FlashInfer JIT-compiles its sampler on first use and fails without a full CUDA toolkit -> use the PyTorch sampler.
export PATH=~/vllm-env/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0
NGPU=1 MAX_SEQS=64 DTYPE="--quantization fp8" MODEL=~/models/main bash scripts/serve_llm.sh
CONFIG=configs/gpu1.yaml bash scripts/run_gpu.sh smoke                     # 20/20 correct on biz_101
CONFIG=configs/gpu1.yaml bash scripts/run_gpu.sh sub300 --concurrency 14   # 10 questions per train shop, scored
CONFIG=configs/gpu1.yaml bash scripts/run_gpu.sh public --concurrency 14
```

- Run **one split at a time** with `--concurrency` ≈ 14: more questions in flight overflow the KV cache, the prefix
  cache hit rate falls from ~85 % to ~40 % and prefill dominates. Measured cost: ~3.6 LLM calls, ~42k prompt and
  ~1.9k completion tokens per question -> ~1.2 questions/min on one 3090 (prefill-bound), i.e. ~14 h for public_test.
- `dma.run` fills abstain for every unanswered id when it stops, so `--resume` after a kill skips everything: keep only
  lines that have a `logs/<id>.json` trace before resuming.
- `scripts/make_sample.py` builds the stratified `data/samples/train_s300` split used by `run_gpu.sh sub300`.
- Server tuning on 1×3090 (`vllm bench serve`, prefix_repetition 8k+3k → 500 tokens, 14 concurrent):
  `--max-num-batched-tokens 16384` is +9 % over the default (now `MAX_BATCHED` default in serve_llm.sh);
  32768 returns HTTP 500; MTP speculative decoding (`qwen3_5_mtp`) is ~40 % *slower* at this batch size and takes
  ~30 % of the KV cache; `--kv-cache-dtype fp8` needs nvcc (absent). Alone on the GPU, sub300 took 98 min (~3 q/min).
- Calibrator without the 40 h CV: `scripts/oof_from_run.py` turns any scored run's traces into calibrator rows
  (out-of-fold as long as the exemplar memory holds no question of those shops), then `scripts/train_calibrator.py`.
  The untrained fallback is kept in `assets/calibrator.fallback.json`.

Fixed during the first real-model runs:
- `semantic/builder.vote`: Qwen3.5-9B omits `tables`/`roles` in BSM refine samples ("keep the proposal"); the vote read
  the omission as "no role" (3 × 1.0 > 1.5) and wiped every role -> 0 views on some shops, answers 0 / unanswerable.
  A sample now votes only on tables it assigned roles in (test `test_bsm_vote_keeps_proposal_when_samples_omit_roles`).
  Train sample PA went 0.14 -> 0.71 on the first ~40 questions.
- `llm/gateway`: a read timeout (long thinking under load) no longer marks the replica unhealthy, and with external
  servers the gateway re-probes `/health` before raising `LLMUnavailable` (one timeout used to abstain the whole run).
- BSM refine samples and rulebook extraction votes run concurrently (same seeds and order -> same result, ~3-5x faster).
- Money rounding (rulebook: money, incl. an average of money, rounds half up to whole đồng): RuleSpec
  `numbers.money_decimals`, money metrics flagged in the catalog, rounded in `Compiler._op_aggregate`.
- Telex typing ("soos bill", "nhiêu ddown") was repaired for the NLU only; the prompt text now shows it retyped in
  Vietnamese (`nlu/typos.retype_telex`), so the solvers stop reading it as revenue.
- Ontology: "thời gian tạo / lập" -> `order_datetime`, so shops with a separate creation timestamp get the shift
  (ca sáng / ca chiều) dimension instead of the `time_of_day` gap.
- sub300 PA: 0.683 (first run) -> 0.750 with the three fixes above.

## 9. Packaging for submission (later)

`Dockerfile` + `predict.sh` + `configs/final.yaml` implement the offline image: weights copied to
`/opt/models/main` (with `SHA256SUMS`), the supervisor in `dma.run` starts one vLLM replica per GPU itself, reads
`/data`, writes `/result/submission.jsonl`, and always leaves a valid submission (abstain-filled) even on crash.
Before building: put the weights in `models/main/`, generate `SHA256SUMS` there, set the base image to a vLLM build
that supports the model, and run `scripts/docker_offline_test.sh`. Details: `../11-docker-deployment.md`.

## 10. Known open items

- biz_105 (train): monthly revenue ~40–80k VND above the answer (< 0.02 %); cause not found yet.
- Calibrator and exemplars are placeholders until step 7 is done with the real model.
- The pipeline's LLM stages (BSM refine, rule extraction votes, plan/code solvers, judge) are covered only by a fake
  LLM in tests; their real accuracy is unknown until 6.3.
