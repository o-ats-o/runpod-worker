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
python3 -c "
from pyannote.audio import Pipeline
print('-> 話者分離モデルのダウンロードを開始します...')
Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token='$HF_TOKEN',
    cache_dir='/app/models'
)
"

echo "\成功！全てのモデルがイメージ内にダウンロードされました！"