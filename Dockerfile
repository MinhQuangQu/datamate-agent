# syntax=docker/dockerfile:1.7
# DataMate Agent — solution image. See 11-docker-deployment.md.
# BTC: replace BASE_IMAGE if the organiser's Docker guideline mandates a base image.
# Pick a CUDA 12.8/12.9 build unless the organiser confirms driver >= 580 (10-setup-guide.md §2).
ARG BASE_IMAGE=vllm/vllm-openai:v0.30.0
FROM ${BASE_IMAGE}

ENV DMA_APP_ROOT=/app DMA_ONTOLOGY=/app/assets/ontology.yaml
ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONHASHSEED=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/hf XDG_CACHE_HOME=/opt/cache TRITON_CACHE_DIR=/opt/cache/triton VLLM_CACHE_ROOT=/opt/cache/vllm \
    PYTHONPATH=/app/src DMA_CONFIG=/app/configs/final.yaml

WORKDIR /app

# 1) app dependencies — never reinstall torch/vllm from the base image
COPY requirements-app.txt /app/
RUN pip install --no-cache-dir -r /app/requirements-app.txt \
 && python -c "import duckdb, rapidfuzz, python_calamine, pydantic, httpx, yaml"

# 2) weights (biggest layer first so code changes do not invalidate it)
COPY models/main /opt/models/main
RUN cd /opt/models/main && sha256sum -c SHA256SUMS

# 3) code, configs, assets, training code, data
COPY src /app/src
COPY configs /app/configs
COPY assets /app/assets
COPY scripts /app/scripts
COPY training /app/training
COPY synth /app/synth
COPY data_train /app/data_train
COPY predict.sh README.md SOLUTION_README.md /app/
RUN chmod +x /app/predict.sh && chmod -R a+rX /app /opt/models && python -m compileall -q /app/src \
 && python -m dma.utils.count_params /opt/models/main --limit 15e9 --json /app/params.json

# 4) JIT caches warmed on an RTX 5090 build host (optional, see §4)
COPY cache/ /opt/cache/
RUN chmod -R a+rwX /opt/cache

ENTRYPOINT []
CMD ["bash", "/app/predict.sh"]
