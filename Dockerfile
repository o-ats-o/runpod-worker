# PyTorchとCUDAがプリインストールされたRunPod公式イメージを使用
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 環境変数を設定
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# ffmpegのようなシステムレベルの依存関係をインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# コンテナ内の作業ディレクトリを設定
WORKDIR /app

# Pythonの依存関係をコピーしてインストール
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# アプリケーションコードをコピー
COPY runpod_handler.py .

# コンテナ起動時に実行するコマンドを定義
CMD ["python", "-u", "runpod_handler.py"]