# =====================================================================
# Runtime & Build Base (CUDA 11.8 + cuDNN 8) to match torch cu118 wheels
# =====================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HUGGING_FACE_HUB_CACHE=/app/models/hf_cache \
    WHISPER_LOCAL_DIR=/app/models/whisper-large-v2 \
    PYANNOTE_CACHE_DIR=/app/models/pyannote \
    HF_HUB_OFFLINE=1

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        build-essential pkg-config \
        git ffmpeg libsndfile1 libsox-dev curl ca-certificates && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/


COPY requirements.txt .
COPY constraints.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu118 \
        -r requirements.txt -c constraints.txt


COPY download_models.py .

RUN --mount=type=secret,id=hf_token \
    export HF_TOKEN="$(cat /run/secrets/hf_token)" && \
    echo "[Bake] HF_HUB_OFFLINE temporarily disabled for download." && \
    HF_HUB_OFFLINE=0 python download_models.py && \
    test -f "$WHISPER_LOCAL_DIR/config.json" && \
    test -f "$PYANNOTE_CACHE_DIR/pipelines/speaker-diarization-3.1/config.yaml" && \
    echo "[Bake] Model bake finished."

ENV HF_HUB_OFFLINE=1

COPY runpod_handler.py .

CMD ["python", "-u", "runpod_handler.py"]