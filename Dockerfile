# PyTorchとCUDAがプリインストールされたRunPod公式イメージを使用
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 環境変数を設定
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    HF_HOME=/app/models/hf_cache

# ffmpegのようなシステムレベルの依存関係をインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# 1. Python依存関係のインストール (変更頻度が低いため先に行い、キャッシュを活用)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# 2. モデル準備スクリプトをコピー
COPY prepare_models.py.

# 3. モデルのダウンロードと設定
# GitHub Actionsからビルド時シークレットをマウントして実行
RUN --mount=type=secret,id=hf_token python prepare_models.py

# 4. アプリケーションコードをコピー
COPY runpod_handler.py .

# コンテナ起動時に実行するコマンドを定義
CMD ["python", "-u", "runpod_handler.py"]