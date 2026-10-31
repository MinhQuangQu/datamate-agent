#!/usr/bin/env bash
# Run the agent against the local vLLM replicas started by scripts/serve_llm.sh (ports 8001, 8002, ...).
#   bash scripts/run_gpu.sh smoke      # 20 questions of biz_101, scored
#   bash scripts/run_gpu.sh sample     # the 12-question stratified sample (data/samples/train_s12), scored
#   bash scripts/run_gpu.sh sub300     # 10 questions per shop (scripts/make_sample.py -> data/samples/train_s300), scored
#   bash scripts/run_gpu.sh train      # full train split (3000 questions), scored per shop
#   bash scripts/run_gpu.sh public     # public_test -> runs/public/submission.jsonl (no answers: upload it)
# Extra args go to dma.run, e.g. `bash scripts/run_gpu.sh train --resume`.
# CONFIG=configs/gpu1.yaml picks another config (default configs/gpu.yaml).
set -uo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-smoke}; shift || true
NGPU=${NGPU:-$(nvidia-smi -L 2>/dev/null | wc -l)}
[ "$NGPU" -ge 1 ] || NGPU=1
EP=$(seq 0 $((NGPU - 1)) | awk '{printf "%shttp://127.0.0.1:%d/v1", (NR > 1 ? "," : ""), 8001 + $1}')
for p in $(seq 8001 $((8000 + NGPU))); do
  curl -sf "http://127.0.0.1:$p/health" >/dev/null || { echo "vLLM not answering on :$p — start scripts/serve_llm.sh first"; exit 1; }
done
PY=.venv/bin/python
CONFIG=${CONFIG:-configs/gpu.yaml}
case "$MODE" in
  smoke)  DATA=data/raw/train;       OUT=runs/smoke;      ARGS=(--only-business biz_101 --limit 20) ;;
  sample) DATA=data/samples/train_s12; OUT=runs/sample;   ARGS=() ;;
  sub300) DATA=data/samples/train_s300; OUT=runs/sub300; ARGS=() ;;
  train)  DATA=data/raw/train;       OUT=runs/train_full; ARGS=() ;;
  public) DATA=data/raw/public_test; OUT=runs/public;     ARGS=() ;;
  *) echo "usage: $0 smoke|sample|sub300|train|public [dma.run args]"; exit 2 ;;
esac
echo "endpoints: $EP  data: $DATA  out: $OUT"
START=$(date +%s)
$PY -m dma.run --config "$CONFIG" --data "$DATA" --result "$OUT" --work "/tmp/dma_$MODE" \
    --endpoints "$EP" "${ARGS[@]}" "$@"
$PY -m dma.io.finalize --data "$DATA" --result "$OUT" || true
echo "wall: $(( $(date +%s) - START ))s"
$PY scripts/validate_submission.py "$OUT/submission.jsonl" "$DATA/questions.jsonl"
if [ -f "$DATA/answers.jsonl" ]; then
  $PY -m dma.eval.scorer --pred "$OUT/submission.jsonl" --gold "$DATA/answers.jsonl" \
      --questions "$DATA/questions.jsonl" --report "$OUT/score.json"
else
  echo "no answers for $DATA: upload $OUT/submission.jsonl to the portal to get the score"
fi
