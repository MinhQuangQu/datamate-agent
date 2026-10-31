#!/usr/bin/env bash
# Start one vLLM OpenAI-compatible server per GPU (data-parallel by hand: ports 8001, 8002, ...).
# Used on dev machines; inside the image the supervisor in dma.run starts the same commands.
# Check every flag against `vllm serve --help` of the pinned vLLM version.
#   MODEL=/models/qwen3.5-9b bash deploy/serve_llm.sh
#   DTYPE="--quantization fp8" VLLM_EXTRA="--reasoning-parser qwen3" bash deploy/serve_llm.sh
#   MAX_LEN=24576 MAX_SEQS=32 GPU_UTIL=0.92 ...   # smaller GPUs
#   MAX_BATCHED: prefill chunk; 16384 measured +9% vs the default on 1x3090 (32768 fails there)
set -euo pipefail
MODEL=${MODEL:-/models/qwen3.5-9b}
NGPU=${NGPU:-$(nvidia-smi -L | wc -l)}
read -r -a DTYPE_ARGS <<< "${DTYPE:---dtype bfloat16}"
EXTRA_ARGS=(--reasoning-parser qwen3 --limit-mm-per-prompt '{"image": 0, "video": 0}')
if [ -n "${VLLM_EXTRA:-}" ]; then read -r -a EXTRA_ARGS <<< "$VLLM_EXTRA"; fi
mkdir -p logs
for ((GPU = 0; GPU < NGPU; GPU++)); do
  PORT=$((8001 + GPU))
  CUDA_VISIBLE_DEVICES=$GPU nohup vllm serve "$MODEL" \
    --served-model-name main --host 127.0.0.1 --port "$PORT" \
    "${DTYPE_ARGS[@]}" --max-model-len "${MAX_LEN:-32768}" --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
    --max-num-seqs "${MAX_SEQS:-64}" --max-num-batched-tokens "${MAX_BATCHED:-16384}" --enable-prefix-caching --seed 0 --generation-config vllm \
    "${EXTRA_ARGS[@]}" > "logs/vllm_gpu${GPU}.log" 2>&1 &
  echo "GPU$GPU -> :$PORT (pid $!)"
done
for ((GPU = 0; GPU < NGPU; GPU++)); do
  PORT=$((8001 + GPU))
  until curl -sf "http://127.0.0.1:$PORT/health" > /dev/null; do sleep 5; done
  echo "replica :$PORT ready"
done
