#!/usr/bin/env bash
# One-shot setup of a GPU server for DataMate Agent (see SETUP_LLM.md). Idempotent: re-run safely.
#   bash scripts/setup_server.sh                      # data + app venv + tests + vLLM venv + model download
#   SKIP_MODEL=1 bash scripts/setup_server.sh         # weights already in $MODEL_DIR
#   MODEL_ID=Qwen/Qwen3-14B bash scripts/setup_server.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
MODEL_DIR=${MODEL_DIR:-$HOME/models/main}
VLLM_ENV=${VLLM_ENV:-$HOME/vllm-env}
PY=${PY:-python3}

echo "== 1/5 data"
mkdir -p data/raw
for s in train public_test; do
  if [ ! -f "data/raw/$s/questions.jsonl" ]; then
    [ -f "../$s.zip" ] || { echo "missing ../$s.zip"; exit 1; }
    unzip -q -o "../$s.zip" -d data/raw
  fi
  echo "  $s: $(wc -l < data/raw/$s/questions.jsonl) questions, $(ls data/raw/$s/data | wc -l) shops"
done

echo "== 2/5 app venv (.venv)"
"$PY" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' || { echo "need Python >= 3.11"; exit 1; }
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -e ".[dev]" httpx charset-normalizer

echo "== 3/5 tests (CPU, fake LLM)"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_no_id_logic.py

echo "== 4/5 vLLM venv ($VLLM_ENV)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv \
  || echo "  WARNING: nvidia-smi not found"
[ -x "$VLLM_ENV/bin/python" ] || "$PY" -m venv "$VLLM_ENV"
"$VLLM_ENV/bin/pip" install -q -U pip
"$VLLM_ENV/bin/pip" install -q -U vllm huggingface_hub
"$VLLM_ENV/bin/python" -c "import vllm, torch; print('  vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.device_count(), 'GPU')"

echo "== 5/5 model $MODEL_ID -> $MODEL_DIR"
if [ "${SKIP_MODEL:-0}" != "1" ]; then
  "$VLLM_ENV/bin/python" - "$MODEL_ID" "$MODEL_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
EOF
fi
.venv/bin/python -m dma.utils.count_params "$MODEL_DIR" --limit 15e9

cat <<EOF

Setup done. Next:
  source $VLLM_ENV/bin/activate && MODEL=$MODEL_DIR bash scripts/serve_llm.sh     # starts 1 replica per GPU
  bash scripts/run_gpu.sh smoke                                                   # 20 questions, scored
  bash scripts/run_gpu.sh train                                                   # 3000 questions, scored
  bash scripts/run_gpu.sh public                                                  # submission for the portal
EOF
