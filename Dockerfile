# =====================================================================
# Runtime & Build Base (CUDA 11.8 + cuDNN 8) to match torch cu118 wheels
# =====================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    # Hugging Face キャッシュ (HF 標準の sha ディレクトリ群)
    HUGGING_FACE_HUB_CACHE=/app/models/hf_cache \
    # Whisper (faster-whisper large-v2) を展開する固定パス
    WHISPER_LOCAL_DIR=/app/models/whisper-large-v2 \
    # pyannote 関連パイプライン用ローカルパス
    PYANNOTE_CACHE_DIR=/app/models/pyannote \
    # デフォルトはオフライン (再取得したいときは実行時に 0 にするか unset)
    HF_HUB_OFFLINE=1

# OS 依存パッケージ
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        build-essential pkg-config \
        git ffmpeg libsndfile1 libsox-dev curl ca-certificates && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --------------------------------------------------------------------------------
# 1. torch (CUDA 11.8 variant) を先に固定インストール (依存のブレを防ぐ)
# --------------------------------------------------------------------------------
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    python -c "import torch;print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())"

# --------------------------------------------------------------------------------
# 2. その他依存 (torch 以外)
# --------------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python - <<'PY'
import torch
print('Post-install check -> Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuDNN:', torch.backends.cudnn.version())
PY

# --------------------------------------------------------------------------------
# 3. モデルダウンロード (HF トークンは BuildKit secret から; トークン自体はレイヤに残さない)
# --------------------------------------------------------------------------------
COPY download_models.py .

# --mount=type=secret,id=hf_token で /run/secrets/hf_token にトークンが供給される
RUN --mount=type=secret,id=hf_token \
    set -euo pipefail; \
    export HF_TOKEN="$(cat /run/secrets/hf_token)"; \
    python download_models.py; \
    test -f "$WHISPER_LOCAL_DIR/config.json" || (echo 'Whisper model missing!' && exit 1); \
    test -f "$PYANNOTE_CACHE_DIR/pipelines/speaker-diarization-3.1/config.yaml" || (echo 'pyannote pipeline missing!' && exit 1); \
    echo 'Model bake finished.'

# --------------------------------------------------------------------------------
# 4. 推論ハンドラ
# --------------------------------------------------------------------------------
COPY runpod_handler.py .

# --------------------------------------------------------------------------------
# 5. 最終コマンド
# --------------------------------------------------------------------------------
CMD ["python", "-u", "runpod_handler.py"]