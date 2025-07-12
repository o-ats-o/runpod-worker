FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# 環境変数の設定
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

# 必要なシステムパッケージをインストール
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip git ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# アプリケーションコードと依存関係をコピー
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY runpod_handler.py .

# デフォルトのコマンド (RunPodワーカーの起動)
CMD ["python", "-u", "runpod_handler.py"]