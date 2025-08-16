#!/bin/bash
# エラーが発生したらスクリプトを停止する
set -e

echo "ステップ1：必要なライブラリをインストール中..."
# ビルド時に必要なライブラリをここで再確認
pip install -U faster-whisper pyannote.audio torch huggingface_hub hf-transfer &> /dev/null && echo "ライブラリのインストールが完了しました。" || echo "ライブラリのインストールに失敗しました。"

echo "ステップ2：シークレットからHugging Faceトークンを設定中..."
# ビルドシークレットからトークンを読み込む
HF_TOKEN=$(cat /run/secrets/hf_token)
export HUGGING_FACE_TOKEN=$HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "エラー：Hugging Faceトークンをシークレットから読み込めませんでした。"
    exit 1
fi

echo "ステップ3-A：Whisperモデルをダウンロード中..."
huggingface-cli download Systran/faster-whisper-large-v2 \
--cache-dir /app/models \
--local-dir /app/models/Systran/faster-whisper-large-v2 \
--local-dir-use-symlinks False

echo "ステップ3-B：話者分離モデルをダウンロード中..."
huggingface-cli download pyannote/speaker-diarization-3.1 --local-dir /app/models/pyannote/speaker-diarization-3.1 --cache-dir /app/models
huggingface-cli download pyannote/segmentation-3.0 --local-dir /app/models/pyannote/segmentation-3.0 --cache-dir /app/models
huggingface-cli download speechbrain/spkrec-ecapa-voxceleb --local-dir /app/models/speechbrain/spkrec-ecapa-voxceleb --cache-dir /app/models

echo "ステップ4：Pyanonteの設定ファイル（config.yaml）をローカルパスに書き換え中..."
CONFIG_PATH="/app/models/pyannote/speaker-diarization-3.1/config.yaml"
sed -i 's|pyannote/segmentation-3.0|/app/models/pyannote/segmentation-3.0|g' $CONFIG_PATH
sed -i 's|speechbrain/spkrec-ecapa-voxceleb|/app/models/speechbrain/spkrec-ecapa-voxceleb|g' $CONFIG_PATH

echo "\成功！全てのモデルがイメージ内にダウンロードされました！"