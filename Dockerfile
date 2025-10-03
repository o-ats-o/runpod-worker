# =====================================================================
# Stage 1: Builder
# このステージでは、ビルドに必要なツールをインストールし、
# Python依存関係を導入し、モデルをダウンロードします。
# =====================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS builder

ARG FORCE_MODEL_REFRESH=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HF_HOME=/app/cache \
    HUGGINGFACE_HUB_CACHE=/app/cache/hub

WORKDIR /app

# ビルドに必要なツールをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 python3-pip python3-distutils python3-venv \
    git curl ca-certificates ffmpeg libsndfile1 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Python依存関係をインストール
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

COPY download_models.py .

# モデルをダウンロードし、設定ファイルを生成
# HF_TOKENはビルド時のシークレットとして渡されます
RUN --mount=type=secret,id=hf_token \
    set -e; \
    export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    HF_HUB_OFFLINE=0 python download_models.py; \
    echo " Model bake and config generation finished."

# =====================================================================
# Stage 2: Final Runtime Image
# このステージでは、builderステージから必要なアーティファクトのみを
# コピーし、軽量でセキュアなランタイムイメージを作成します。
# =====================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HF_HUB_OFFLINE=1 \
    # huggingface系のライブラリが参照するキャッシュパス
    HF_HOME=/app/models \
    # pyannote.audioが独自に参照するキャッシュパス
    PYANNOTE_CACHE=/app/models \
    WHISPER_LOCAL_DIR=/app/models/whisper-large-v2 \
    RESEMBLE_ENHANCE_RUN_DIR=/app/models/resemble-enhance/enhancer_stage2

WORKDIR /app

# ランタイムに必要なOSパッケージのみをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 libsndfile1 ffmpeg && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Builderステージから必要なファイルのみをコピー
COPY --from=builder /usr/local/lib/python3.10/dist-packages /usr/local/lib/python3.10/dist-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/models /app/models
COPY --from=builder /app/diarization_config.yaml /app/diarization_config.yaml
COPY runpod_handler.py .

CMD ["python", "-u", "runpod_handler.py"]