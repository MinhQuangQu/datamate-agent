#!/usr/bin/env bash
# DataMate Agent — entrypoint called by the organiser.
# Reads /data (read-only), writes /result/submission.jsonl and /result/logs/.
# No `set -e`: the finalize step must always run so a valid submission is left behind.
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 PYTHONHASHSEED=0
export PYTHONPATH=${PYTHONPATH:-/app/src}
DATA_DIR=${DATA_DIR:-/data}
RESULT_DIR=${RESULT_DIR:-/result}
CONFIG=${DMA_CONFIG:-/app/configs/final.yaml}
mkdir -p "$RESULT_DIR/logs" /tmp/dma
LOG="$RESULT_DIR/logs/predict.log"
START=$(date +%s)
export DMA_START_EPOCH=$START   # the run clock (soft/hard deadlines) counts from here, incl. model load
{
  echo "[predict] start $(date -Is)"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
  python -c "import torch, vllm; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)" || true
  df -h /tmp "$RESULT_DIR" || true
} >> "$LOG" 2>&1

EXTRA=()
if [ "${DMA_WARMUP_ONLY:-0}" = "1" ]; then EXTRA+=(--warmup-only); fi
python -m dma.run --config "$CONFIG" --data "$DATA_DIR" --result "$RESULT_DIR" "${EXTRA[@]}" >> "$LOG" 2>&1
RC=$?

# safety net: one valid line per question id (abstain for missing), atomic rewrite
if [ "${DMA_WARMUP_ONLY:-0}" != "1" ]; then
  python -m dma.io.finalize --data "$DATA_DIR" --result "$RESULT_DIR" >> "$LOG" 2>&1 || true
fi
echo "[predict] end rc=$RC elapsed=$(( $(date +%s) - START ))s" >> "$LOG"
exit 0
