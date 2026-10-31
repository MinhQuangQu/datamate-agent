# DataMate Agent (DMA)

Agent offline trả lời câu hỏi tiếng Việt trên dữ liệu kinh doanh của cửa hàng nhỏ (CSV / XLSX / SQLite) cho
**Zalo AI Challenge 2026 — DataMate**. Mỗi câu chỉ có một đáp án hoặc `abstain`. Điểm là Penalized Accuracy:
PA = (đúng − sai) / tổng số câu, nên trả lời sai còn tệ hơn bỏ qua.

| Phiên bản | Train sub300 (PA) | Public (portal) | Thay đổi chính | Doc |
|---|---|---|---|---|
| v1 | 0.683 | 0.14 | Kiến trúc gốc: semantic layer → 2 solver → vote/judge → calibrator | [docs/v1-baseline.md](docs/v1-baseline.md) |
| v2 | 0.750 | 0.17 | Làm tròn tiền, Telex, ontology, calibrator đã huấn luyện | [docs/v2-train-fixes.md](docs/v2-train-fixes.md) |
| v3 | 0.807 | 0.425 | Sửa tầng đọc dữ liệu tất định cho kiểu file mới, bộ phân giải thời gian | [docs/v3-deterministic-layer.md](docs/v3-deterministic-layer.md) |
| v4 | 0.870 | 0.433 | Verifier agent khi bất đồng, sandbox giữ dữ liệu khi restart, DMQL `having` | [docs/v4-verifier-agent.md](docs/v4-verifier-agent.md) |
| v5 | chưa đo | chưa nộp | Lọc theo thuộc tính entity, nhóm hàng từ bảng phụ, bỏ cột hằng | [docs/v5-entity-attributes.md](docs/v5-entity-attributes.md) |

Đọc [docs/README.md](docs/README.md) để biết kiến trúc thay đổi qua từng bản.

---

## 1. Kiến trúc tóm tắt

```
split/ (questions.jsonl, rulebook.md, calendar.csv, data/<shop>/…)
  │
  ├─ 1 lần / split   RuleSpec: rulebook → cấu hình có cấu trúc (diff với rulebook train, LLM trích phần đổi)
  ├─ 1 lần / shop    BKP (Business Knowledge Pack):
  │                    ingest/   đọc file → DuckDB (raw/typed), suy kiểu số/ngày kiểu VN, gộp sheet, bỏ dòng tổng
  │                    profile/  thống kê cột, khoá, FK
  │                    semantic/ gán vai trò cột (ontology + heuristic + LLM vote) → BSM → view sem.* áp luật
  │                              metric catalog, value index, Business Card (đưa vào prompt)
  └─ mỗi câu         pipeline.answer_question
                       nlu/        chuẩn hoá, Telex, thời gian (âm lịch, Tết, tuần…), link thực thể, gap
                       Solver A    plan: LLM viết DMQL (JSON) → compiler.py → SQL tất định
                       Solver B    code: LLM viết Python nhiều bước trong sandbox, dùng toolkit `dm`
                       arbitration A==B → agree_ab; khác → thêm mẫu → vote → verifier agent → judge
                       calibrator  P(đúng) → trả lời hoặc abstain
                       formatter   chuẩn hoá theo type (number/percent/date/list/table…)
```

LLM chỉ được gọi qua endpoint OpenAI-compatible `/v1/chat/completions`, vì vậy đổi model chỉ là đổi cách serve,
không phải sửa code. Model hiện dùng là **Qwen3.5-9B** (giới hạn cuộc thi: ≤ 15B tham số), serve bằng **vLLM**.

---

## 2. Yêu cầu

| Thứ | Tối thiểu | Ghi chú |
|---|---|---|
| OS | Linux x86_64 | đã chạy trên Ubuntu, driver 580 / CUDA 13 |
| Python | ≥ 3.11 cho app; 3.12 cho venv vLLM | hai venv riêng |
| GPU | 1× 24 GB (RTX 3090/4090) | 3090 phải serve FP8 weight-only; máy đích là 2× RTX 5090 |
| Đĩa | ~30 GB | model ~19 GB, venv vLLM ~8 GB, dữ liệu ~0.3 GB |
| Dữ liệu | `train.zip`, `public_test.zip` | tải từ portal cuộc thi, **không có trong repo** |

Không có GPU vẫn chạy được toàn bộ test và pipeline với LLM giả (mục 4).

---

## 3. Setup từng bước

### 3.1 Clone và đặt dữ liệu

```bash
git clone <url-repo> datamate-agent
cd datamate-agent
# đặt hai file zip của BTC ở thư mục CHA của repo:
#   ../train.zip  ../public_test.zip
mkdir -p data/raw
unzip -q ../train.zip -d data/raw
unzip -q ../public_test.zip -d data/raw
ls data/raw/train data/raw/public_test      # mỗi split: questions.jsonl, rulebook.md, calendar.csv, data/
```

`data/`, `runs/`, `models/`, `logs/` nằm trong `.gitignore`. Không commit dữ liệu hay kết quả chạy.

### 3.2 Venv cho app (CPU)

```bash
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -e ".[dev]" httpx charset-normalizer
.venv/bin/python -m pytest -q                    # ~40 s, phải pass hết (90 test)
.venv/bin/python scripts/check_no_id_logic.py    # kiểm không có logic theo ID shop/câu
```

### 3.3 Venv cho vLLM và tải model (GPU)

```bash
python3.12 -m venv ~/vllm-env
~/vllm-env/bin/pip install -U pip vllm huggingface_hub
~/vllm-env/bin/python -c "import vllm, torch; print(vllm.__version__, torch.cuda.device_count())"

~/vllm-env/bin/python -c "import os; from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='Qwen/Qwen3.5-9B', local_dir=os.path.expanduser('~/models/main'))"
.venv/bin/python -m dma.utils.count_params ~/models/main --limit 15e9   # kiểm giới hạn 15B
```

Hoặc chạy một lệnh làm cả 3.1–3.3: `bash scripts/setup_server.sh` (biến `MODEL_ID`, `MODEL_DIR`, `VLLM_ENV`,
`SKIP_MODEL=1`).

### 3.4 Serve model

Mỗi GPU một replica, cổng 8001, 8002, …

```bash
source ~/vllm-env/bin/activate

# 2 GPU ≥ 32 GB (BF16):
MODEL=~/models/main bash scripts/serve_llm.sh

# 1 GPU 24 GB (RTX 3090) — cấu hình đang dùng để ra mọi kết quả trong bảng trên:
VLLM_USE_FLASHINFER_SAMPLER=0 NGPU=1 MAX_SEQS=64 DTYPE="--quantization fp8" \
  MODEL=~/models/main bash scripts/serve_llm.sh

curl -s http://127.0.0.1:8001/health && echo OK
```

- BF16 9B chiếm 19.3 GB nên trên 24 GB không còn KV cache. FP8 weight-only (Marlin trên Ampere) để lại khoảng 255k token KV.
- `VLLM_USE_FLASHINFER_SAMPLER=0` là bắt buộc khi máy không có CUDA toolkit đầy đủ, vì FlashInfer JIT sampler cần `nvcc`.
- `--max-num-batched-tokens 16384` (mặc định trong script) nhanh hơn khoảng 9% trên 3090. 32768 gây lỗi 500. MTP speculative decoding chậm hơn.
- Log của server nằm ở `logs/vllm_gpu<N>.log`.

Chi tiết và cách xử lý sự cố: [SETUP_LLM.md](SETUP_LLM.md) (§8c cho 1× 3090, §8b cho API hosted như Groq).

### 3.5 Chạy

```bash
# chọn config: configs/gpu1.yaml (1 GPU) hoặc configs/gpu.yaml (2 GPU)
CONFIG=configs/gpu1.yaml bash scripts/run_gpu.sh smoke     # 20 câu của biz_101, có chấm điểm (~10 phút)
CONFIG=configs/gpu1.yaml bash scripts/run_gpu.sh public    # 1000 câu public → runs/public/submission.jsonl
```

Bộ đo nhanh trên train (10 câu × 30 shop, có đáp án), dùng để so sánh mọi thay đổi:

```bash
.venv/bin/python scripts/make_sample.py --split data/raw/train --per-business 10 --out data/samples/train_s300
CONFIG=configs/gpu1.yaml bash scripts/run_gpu.sh sub300    # ~100 phút trên 1× 3090, ghi runs/sub300/score.json
```

Gọi trực tiếp (khi cần đổi thư mục kết quả, tiếp tục run dở, chạy một vài shop):

```bash
.venv/bin/python -m dma.run --config configs/gpu1.yaml \
  --data data/raw/public_test --result runs/v6/public --work /tmp/dma_v6pub \
  --endpoints http://127.0.0.1:8001/v1 --concurrency 14 \
  [--only-business dev_003,dev_006] [--limit 20] [--resume]
```

Kiểm định dạng và chấm điểm:

```bash
.venv/bin/python scripts/validate_submission.py runs/v6/public/submission.jsonl data/raw/public_test/questions.jsonl
.venv/bin/python -m dma.eval.scorer --pred runs/sub300/submission.jsonl \
  --gold data/samples/train_s300/answers.jsonl --questions data/samples/train_s300/questions.jsonl --report runs/sub300/score.json
```

Ghép kết quả chạy lại vài shop vào một file nộp: `scripts/merge_submissions.py --base … --over … --only-business …`.

Tốc độ tham khảo trên 1× 3090: khoảng 3–8 câu/phút khi chạy một mình trên GPU. Public 1000 câu mất 2.5–4 giờ.
Mỗi lần chỉ nên chạy một split, vì hai run chạy song song sẽ chia GPU và làm tăng số câu timeout.

---

## 4. Phát triển không cần GPU

```bash
make test                 # unit + integration + e2e với FakeBackend
make dev-fake             # sinh split tổng hợp (synth/) rồi chạy cả pipeline với LLM giả và chấm điểm
```

FakeBackend trả lời theo luật từ khoá. Điểm chạy với nó chỉ đo các tầng tất định (đọc file, view, compiler,
thời gian, định dạng), không đo độ chính xác của model.

**Kiểm tầng dữ liệu tất định (không LLM).** Đây là cách đã tìm ra phần lớn lỗi của v3–v5:

```bash
W=/tmp/dma_train
.venv/bin/python -m dma.run --config configs/dev.yaml --data data/raw/train --result runs/build_train --work $W --no-llm
.venv/bin/python scripts/build_one.py --split data/raw/train --work $W biz_106 biz_124     # build lại vài shop
.venv/bin/python scripts/probe_deterministic.py --split data/raw/train --work $W --out runs/probe.jsonl
```

Sau đó mở `$W/<shop>/bkp.duckdb` và so các view `sem.*` (doanh thu và số đơn theo tháng, category, loại khách)
với một phép tính pandas viết riêng trên file gốc. Nếu lệch thì lỗi nằm ở tầng chung, và lỗi này làm **cả hai solver
cùng sai** dù vẫn được gắn agree_ab.

---

## 5. Quy tắc khi cải tiến

1. **Không có logic theo ID shop hoặc ID câu** (`scripts/check_no_id_logic.py` phải sạch). Không đưa câu public vào
   exemplar hay dữ liệu huấn luyện. Private test sẽ có shop, kiểu file và luật mới, nên sửa phải là năng lực
   tổng quát của agent (tool, kiểm chứng, định nghĩa), không phải vá theo đề.
2. **Tìm lỗi theo lớp.** Đọc trace (`runs/<run>/logs/<qid>.json`: candidates, arbitration, verify,
   failed_trajectories) để tìm *kiểu* lỗi, sửa kiểu lỗi đó, rồi đo lại trên train sub300.
3. **Kiểm hồi quy tầng tất định** bằng cách build lại 30 shop train với `--no-llm` và diff view với bản build trước.
   Chỉ những cột bạn định sửa được phép đổi.
4. **Đo trước khi nộp:** chạy `make test`, sau đó sub300. Ghi PA và số đúng/sai/abstain vào doc của phiên bản mới
   trong `docs/`.
5. Mỗi phiên bản mới thì thêm `docs/vN-<tên>.md` theo mẫu các doc có sẵn và cập nhật bảng ở đầu README.

---

## 6. Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `src/dma/io/` | tìm split, đọc câu hỏi, `finalize` (đảm bảo mỗi id có đúng một dòng hợp lệ, kể cả khi bị kill) |
| `src/dma/ingest/` | đọc CSV (encoding/delimiter), XLSX (merged cell, sheet ẩn, header 2 dòng), SQLite; tách block, bỏ dòng tổng, suy kiểu cột → DuckDB |
| `src/dma/profile/` | thống kê, khoá, FK, đồng nhất số học, Data Card |
| `src/dma/semantic/` | `ontology.yaml` + `heuristics.py` (vai trò cột), `builder.py` (LLM refine + vote), `views.py` (view `sem.*`), `catalog.py` (metric, gap), `card.py` (Business Card), `value_index` |
| `src/dma/rules/` | parse rulebook, diff với rulebook train, trích luật → RuleSpec |
| `src/dma/timeparse/` | âm lịch, `calendar.csv`, phân giải thời gian tiếng Việt |
| `src/dma/nlu/` | chuẩn hoá, Telex, khái niệm/gap, gợi ý type, link thực thể |
| `src/dma/solvers/plan/` | DMQL schema, `compiler.py` (DMQL → SQL), plan solver + repair |
| `src/dma/solvers/code/` | `sandbox.py` (kernel cách ly, timeout, rebase), `toolkit.py` (`dm.*`, dùng lại compiler), `agent.py` (CodeAct) |
| `src/dma/verify/` | canonical form, kiểm S1–S11, `arbiter.py` (vote, verifier agent, judge), calibrator |
| `src/dma/llm/` | gateway (nhiều replica, retry), supervisor vLLM, `prompts.py`, FakeBackend |
| `src/dma/pipeline.py`, `run.py` | một câu end-to-end; orchestrator, deadline, log, manifest |
| `src/dma/config.py`, `configs/*.yaml` | mặc định và profile (`dev`, `gpu`, `gpu1`, `final`, `groq`) |
| `assets/` | ontology, rulebook/RuleSpec train tham chiếu, exemplars, calibrator |
| `scripts/` | serve, run, sample, validate, merge, build_one, probe, calibrator, CV |
| `synth/` | sinh split tổng hợp + oracle pandas độc lập |
| `tests/` | unit, integration, e2e (FakeBackend) |
| `submissions/` | các file nộp lên portal (v1–v5), làm mốc so sánh |
| `docs/` | lịch sử kiến trúc từng phiên bản |
| `Dockerfile`, `predict.sh`, `SOLUTION_README.md` | đóng gói image nộp cuối (2× 5090, offline) |

### Các nút cấu hình hay chỉnh (`configs/*.yaml`)

| Khoá | Ý nghĩa |
|---|---|
| `run.max_concurrency`, `--concurrency` | số câu chạy song song (1× 3090: 14) |
| `run.question_timeout_s` | timeout mỗi câu |
| `llm.endpoints`, `served_name` | địa chỉ vLLM; `served_name` phải trùng `--served-model-name` |
| `llm.thinking.<stage>`, `llm.max_tokens.<stage>` | bật/tắt thinking và ngân sách token theo công đoạn (rule, bsm, plan, code_first, code_step, verify_first, verify_step, judge) |
| `llm.prompt_rulebook` | `full` (cả rulebook trong prompt) hoặc `relevant` (chỉ các mục liên quan, ít token hơn) |
| `arbitration.verify` | bật verifier agent khi solver bất đồng (v4+) |
| `semantic.bsm_samples` | số mẫu LLM khi gán vai trò cột |

---

## 7. Bảo đảm được test kiểm

- Mỗi id câu hỏi có đúng một dòng hợp lệ, kể cả sau `kill -9`. File nộp < 5 MB.
- Hai lần chạy cho cùng một file nộp: seed sinh từ nội dung, thứ tự và tie-break tất định.
- Không có LLM hoặc vLLM thì vẫn ra file hợp lệ (toàn abstain), exit 0.
- Sandbox chặn subprocess, socket và việc ghi ra ngoài thư mục làm việc. `/data` chỉ được mở read-only.
