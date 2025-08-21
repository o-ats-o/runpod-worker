# =====================================================================
# Runtime & Build Base (CUDA 11.8 + cuDNN 8) to match torch cu118 wheels
# =====================================================================
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ARG FORCE_MODEL_REFRESH=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HF_HOME=/app/hf_home \
    HUGGINGFACE_HUB_CACHE=/app/hf_home/hub \
    WHISPER_LOCAL_DIR=/app/models/whisper-large-v2 \
    PYANNOTE_CACHE_DIR=/app/models/pyannote

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-distutils python3-venv \
        git ffmpeg libsndfile1 libsox-dev curl ca-certificates && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
COPY constraints.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r requirements.txt -c constraints.txt && \
        

COPY download_models.py .

RUN --mount=type=secret,id=hf_token \
    set -e; \
    export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    if [ "$FORCE_MODEL_REFRESH" = "true" ] || [ ! -e "$WHISPER_LOCAL_DIR/config.json" ] || [ ! -e "$PYANNOTE_CACHE_DIR/pipeline_snapshot/config.yaml" ]; then \
        echo "[Bake] Starting model download (FORCE_MODEL_REFRESH=$FORCE_MODEL_REFRESH)"; \
        HF_HUB_OFFLINE=0 python download_models.py; \
    else \
        echo "[Bake] Skip model download (already present)"; \
    fi; \
    echo "[Bake] Listing baked artifacts:"; \
    find /app/models -maxdepth 4 -type f -name 'config.yaml' -o -name 'model.bin' | sed 's/^/[Bake] /'; \
    echo '[Bake] Model bake finished.'

ENV HF_HUB_OFFLINE=1

COPY runpod_handler.py .

CMD ["python", "-u", "runpod_handler.py"]