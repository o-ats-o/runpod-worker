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

echo "ステップ4：Pyanonteの設定ファイル（config.yaml）をローカルパスに書き換え中..."

# findコマンドで関連ファイルの絶対パスを確実に取得する
# grepが何も見つけなくてもスクリプトが停止しないように `|| true` を追加
CONFIG_PATH=$(find /app/models -type f -name "config.yaml" | grep "pyannote/speaker-diarization" || true)
SPEECHBRAIN_PATH=$(find /app/models -type d -name "*speechbrain*spkrec-ecapa-voxceleb*" || true)
SEGMENTATION_PATH=$(find /app/models -type d -name "*pyannote*segmentation*" || true)

# パスが見つかったかどうかのチェックを追加し、デバッグしやすくする
if [ -z "$CONFIG_PATH" ]; then
    echo "致命的エラー: config.yaml が見つかりませんでした。ビルドを中止します。"
    echo "--- /app/models ディレクトリの構造 ---"
    ls -R /app/models # デバッグ用にディレクトリ構造を全て出力
    exit 1
fi

echo "発見したパス:"
echo "  - Config: $CONFIG_PATH"
echo "  - Segmentation: $SEGMENTATION_PATH"
echo "  - Speechbrain: $SPEECHBRAIN_PATH"

# sedコマンドで、config.yaml内のモデルパスを、findで見つけた絶対パスに書き換える
# 区切り文字を'/'から'#'に変更し、パス内の'/'がエラーを引き起こさないようにする
sed -i "s#pyannote/segmentation-3.0#$SEGMENTATION_PATH#g" "$CONFIG_PATH"
sed -i "s#speechbrain/spkrec-ecapa-voxceleb#$SPEECHBRAIN_PATH#g" "$CONFIG_PATH"

echo "\成功！全てのモデルがイメージ内にダウンロードされました！"