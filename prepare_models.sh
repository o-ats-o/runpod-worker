#!/bin/bash
# エラーが発生した場合や未定義の変数があった場合に即座にスクリプトを停止する
set -euo pipefail

echo "ステップ1：必要なライブラリをインストール中..."
pip install -q -U faster-whisper pyannote.audio==3.1.1 torch huggingface_hub hf-transfer
echo "ライブラリのインストールが完了しました。"

echo "ステップ2：シークレットからHugging Faceトークンを設定中..."
HF_TOKEN=$(cat /run/secrets/hf_token)
export HUGGING_FACE_TOKEN=$HF_TOKEN

if [ -z "$HF_TOKEN" ]; then
    echo "致命的エラー：Hugging Faceトークンをシークレットから読み込めませんでした。"
    exit 1
fi
echo "Hugging Faceトークンが設定されました。"

MODELS_DIR="/app/models"

echo "ステップ3-A：Whisperモデルをダウンロード中..."
hf download Systran/faster-whisper-large-v2 \
  --cache-dir $MODELS_DIR \
  --local-dir $MODELS_DIR/Systran/faster-whisper-large-v2 \
  --local-dir-use-symlinks False
echo "Whisperモデルのダウンロードが完了しました。"


echo "ステップ3-B：話者分離モデルと依存モデルをダウンロード中..."
python3 -c "
from huggingface_hub import snapshot_download
import os

hf_token = os.environ.get('HUGGING_FACE_TOKEN')
models_dir = '$MODELS_DIR'

# 1. Pyannote本体 (speaker-diarization-3.1) のダウンロード
print('-> 話者分離パイプライン (diarization-3.1) をダウンロードします...')
snapshot_download(
    'pyannote/speaker-diarization-3.1',
    local_dir=f'{models_dir}/pyannote-diarization-3.1',
    local_dir_use_symlinks=False,
    token=hf_token,
    allow_patterns=['*.yaml', '*.bin', '*.onnx', '*.model'] # 必要なファイルのみダウンロード
)

# 2. 依存モデル① (セグメンテーション) のダウンロード
print('-> セグメンテーションモデル (segmentation-3.0) をダウンロードします...')
snapshot_download(
    'pyannote/segmentation-3.0',
    local_dir=f'{models_dir}/pyannote-segmentation-3.0',
    local_dir_use_symlinks=False,
    token=hf_token,
    allow_patterns=['*.yaml', '*.bin', '*.onnx', '*.model']
)

# 3. 依存モデル② (埋め込み) のダウンロード
# speaker-diarization-3.1のconfig.yamlで指定されている埋め込みモデル
print('-> 埋め込みモデル (wespeaker-voxceleb-resnet34-LM) をダウンロードします...')
snapshot_download(
    'pyannote/wespeaker-voxceleb-resnet34-LM',
    local_dir=f'{models_dir}/pyannote-embedding-wespeaker',
    local_dir_use_symlinks=False,
    token=hf_token,
    allow_patterns=['*.yaml', '*.bin', '*.onnx', '*.model']
)
"
echo "話者分離関連モデルのダウンロードが完了しました。"


echo "ステップ4：Pyanonteの設定ファイル（config.yaml）をローカルパスに書き換え中..."

CONFIG_PATH="$MODELS_DIR/pyannote-diarization-3.1/config.yaml"
SEGMENTATION_PATH="$MODELS_DIR/pyannote-segmentation-3.0"
EMBEDDING_PATH="$MODELS_DIR/pyannote-embedding-wespeaker"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "致命的エラー: config.yaml が見つかりませんでした: ${CONFIG_PATH}"
    echo "--- ${MODELS_DIR} ディレクトリの構造 ---"
    ls -R "${MODELS_DIR}"
    exit 1
fi

echo "発見したパス:"
echo "  - Config: ${CONFIG_PATH}"
echo "  - Segmentation: ${SEGMENTATION_PATH}"
echo "  - Embedding: ${EMBEDDING_PATH}"

echo "-> config.yamlを更新しています..."
sed -i "s#pyannote/segmentation-3.0#${SEGMENTATION_PATH}#g" "${CONFIG_PATH}"
sed -i "s#pyannote/wespeaker-voxceleb-resnet34-LM#${EMBEDDING_PATH}#g" "${CONFIG_PATH}"
echo "config.yamlの更新が完了しました。"

echo ""
echo "✅ 成功！全てのモデルがイメージ内にダウンロードされ、設定が更新されました！"