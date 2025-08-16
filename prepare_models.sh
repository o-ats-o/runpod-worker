#!/bin/bash
# エラーが発生したらスクリプトを停止する
set -e

echo "ステップ1：必要なライブラリをインストール中..."
# ビルド時に必要なライブラリをここで再確認
pip install -U faster-whisper pyannote.audio torch huggingface_hub hf-transfer &> /dev/null && echo "ライブラリのインストールが完了しました。" || echo "ライブラリのインストールに失敗しました。"

echo "ステップ2：シークレットからHugging Faceトークンを設定中..."
# ビルドシークレットからトークンを読み込む
HF_TOKEN=$(cat /run/secrets/hf_token)
export HF_TOKEN
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"

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
token = os.getenv('HUGGING_FACE_TOKEN')
Pipeline.from_pretrained(
    'pyannote/speaker-diarization-3.1',
    use_auth_token=token,
    cache_dir='/app/models'
)
"

echo "ステップ4：Pyanonteの設定ファイル（config.yaml）をローカルパスに書き換え中..."
CONFIG_PATH=$(find /app/models -name "config.yaml" | grep "pyannote/speaker-diarization-3.1")
echo "発見したconfig.yamlのパス: $CONFIG_PATH"
SPEECHBRAIN_PATH=$(find /app/models -type d -name "*speechbrain*spkrec-ecapa-voxceleb*")
echo "発見したspeechbrainモデルのパス: $SPEECHBRAIN_PATH"
SEGMENTATION_PATH=$(find /app/models -type d -name "*pyannote*segmentation*")
echo "発見したsegmentationモデルのパス: $SEGMENTATION_PATH"
sed -i \"s|pyannote/segmentation-3.0|$SEGMENTATION_PATH|g\" $CONFIG_PATH
sed -i \"s|speechbrain/spkrec-ecapa-voxceleb|$SPEECHBRAIN_PATH|g\" $CONFIG_PATH

echo "\成功！全てのモデルがイメージ内にダウンロードされました！"