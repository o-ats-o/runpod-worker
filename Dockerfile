FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python依存関係のインストール
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# シェルスクリプトをコピーし、実行権限を付与
COPY prepare_models.sh .
RUN chmod +x ./prepare_models.sh

# Hugging Faceライブラリにキャッシュの場所を教える
ENV HUGGING_FACE_HUB_CACHE=/app/models

# モデル準備スクリプトを実行
# hf-transferライブラリが高速ダウンロードのためにこの環境変数を参照します
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN --mount=type=secret,id=hf_token ./prepare_models.sh

# 最後にハンドラコードをコピー
COPY runpod_handler.py .

# コンテナ起動時のコマンド
CMD ["python", "-u", "runpod_handler.py"]