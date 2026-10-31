#!/usr/bin/env bash
# Offline acceptance run of the solution image (T1/T2/T8/T10 of 11-docker-deployment.md §5).
#   bash deploy/docker_offline_test.sh IMAGE SPLIT_DIR [OUT_DIR] [extra docker args...]
set -euo pipefail
IMG=$1; SPLIT=$(realpath "$2"); OUT=${3:-/tmp/dma_docker_test}; shift $(( $# < 3 ? $# : 3 ))
rm -rf "$OUT" && mkdir -p "$OUT"
/usr/bin/time -v docker run --rm --gpus all --network none "$@" \
  -v "$SPLIT":/data:ro -v "$OUT":/result "$IMG" 2> "$OUT.time.log"
python scripts/validate_submission.py "$OUT/submission.jsonl" "$SPLIT/questions.jsonl"
if [ -f "$SPLIT/answers.jsonl" ]; then
  python -m dma.eval.scorer --pred "$OUT/submission.jsonl" --gold "$SPLIT/answers.jsonl"
fi
grep -E "Elapsed \(wall clock\)|Maximum resident" "$OUT.time.log" || true
