#!/bin/bash
# エラー発生時に即座にスクリプトを停止し、未定義変数の使用をエラーとする堅牢な設定
set -euo pipefail

echo "ステップ1：シークレットからHugging Faceトークンを設定中..."
# ビルドシークレットからトークンを読み込む
HF_TOKEN_FILE="/run/secrets/hf_token"
if [ ! -f "$HF_TOKEN_FILE" ]; then
    echo "エラー：Hugging Faceトークンファイルが見つかりません: $HF_TOKEN_FILE"
    exit 1
fi
HF_TOKEN=$(cat "$HF_TOKEN_FILE")
if [ -z "$HF_TOKEN" ]; then
    echo "エラー：Hugging Faceトークンをシークレットから読み込めませんでした。"
    exit 1
fi

export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
echo "Hugging Faceトークンが設定されました。"


echo "ステップ2-A：Whisperモデルをダウンロード中..."
hf download Systran/faster-whisper-large-v2 \
--cache-dir /app/models \
--local-dir /app/models/Systran/faster-whisper-large-v2

echo "ステップ2-B & 3：話者分離モデルのダウンロードと設定ファイルの動的書き換え..."
python3 -c "
import os
import yaml
from huggingface_hub import hf_hub_download, snapshot_download
from pathlib import Path

# --- 変数定義 ---
CACHE_DIR = '/app/models'
HF_TOKEN = os.getenv('HUGGING_FACE_HUB_TOKEN')

# パイプラインと、その設定ファイル（config.yaml）で参照される依存モデルのリポジトリID
PYANNOTE_PIPELINE_REPO = 'pyannote/speaker-diarization-3.1'
SEGMENTATION_REPO = 'pyannote/segmentation-3.0'
EMBEDDING_REPO = 'speechbrain/spkrec-ecapa-voxceleb'

print('-> 依存モデルと設定ファイルをダウンロードします...')

# 1. 必要なファイルを確定的にダウンロードし、ローカルの絶対パスを取得
#    - config.yaml: hf_hub_downloadでファイル単体をダウンロードし、そのパスを直接取得
#    - 依存モデル: snapshot_downloadでリポジトリ全体をダウンロードし、そのディレクトリパスを取得
try:
    config_path = hf_hub_download(
        repo_id=PYANNOTE_PIPELINE_REPO,
        filename='config.yaml',
        use_auth_token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )
    print(f'   - config.yamlのパスを取得: {config_path}')

    segmentation_path = snapshot_download(
        repo_id=SEGMENTATION_REPO,
        use_auth_token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )
    print(f'   - segmentationモデルのパスを取得: {segmentation_path}')

    embedding_path = snapshot_download(
        repo_id=EMBEDDING_REPO,
        use_auth_token=HF_TOKEN,
        cache_dir=CACHE_DIR
    )
    print(f'   - embeddingモデルのパスを取得: {embedding_path}')

except Exception as e:
    print(f'モデルのダウンロード中にエラーが発生しました: {e}')
    # デバッグ用にディレクトリ構造を出力
    print('--- /app/models ディレクトリの構造 ---')
    for p in Path(CACHE_DIR).rglob('*'):
        print(p)
    exit(1)


# 2. PyYAMLを使ってconfig.yamlを安全に読み込み、内容を書き換える
print('-> config.yamlを更新しています...')
with open(config_path, 'r') as f:
    config_data = yaml.safe_load(f)

# segmentationとembeddingモデルのパスを、ダウンロードした絶対パスに書き換える
# config.yamlの構造に合わせてキーを指定 (例: pipeline.params.segmentation)
# 実際のキーはconfig.yamlの中身を確認してください
config_data['pipeline']['params']['segmentation'] = segmentation_path
config_data['pipeline']['params']['embedding'] = embedding_path

# 3. 変更内容をファイルに書き戻す
with open(config_path, 'w') as f:
    # default_flow_style=Falseで可読性の高いブロック形式を維持
    yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

print('   - config.yamlの更新が完了しました。')

"

echo "✅ 成功！全てのモデルがイメージ内にダウンロードされ、設定が更新されました！"