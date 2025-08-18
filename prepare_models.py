import os
import yaml
from huggingface_hub import snapshot_download
from pathlib import Path

# --- 設定 ---
# モデルを保存するコンテナ内のベースディレクトリ
MODELS_BASE_DIR = Path("/app/models")

# ダウンロードするHugging Faceリポジトリのリスト
REPOSITORIES = {
    "faster-whisper-large-v2": "Systran/faster-whisper-large-v2",
    "pyannote-diarization-3.1": "pyannote/speaker-diarization-3.1",
    "pyannote-segmentation-3.0": "pyannote/segmentation-3.0",
    "pyannote-wespeaker-voxceleb-resnet34-LM": "pyannote/wespeaker-voxceleb-resnet34-LM",
}

def get_hf_token():
    """ビルド時シークレットまたは環境変数からHugging Faceトークンを取得"""
    try:
        # Dockerビルド時にマウントされたシークレットファイルから読み込む
        with open('/run/secrets/hf_token', 'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("INFO: /run/secrets/hf_token not found. Falling back to HUGGING_FACE_TOKEN env var.")
        return os.getenv("HUGGING_FACE_TOKEN")

def download_models(token):
    """定義されたリポジトリから全てのモデルをダウンロード"""
    for local_name, repo_id in REPOSITORIES.items():
        print(f"--- Downloading {repo_id} to {local_name} ---")
        target_path = MODELS_BASE_DIR / local_name
        target_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_path),
            local_dir_use_symlinks=False,
            token=token,
            resume_download=True,
        )
        print(f"--- Finished downloading {repo_id} ---")

def patch_pyannote_config():
    """pyannoteのconfig.yamlを編集してローカルパスを使用するように変更"""
    diar_pipeline_dir = MODELS_BASE_DIR / "pyannote-diarization-3.1"
    config_path = diar_pipeline_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"ERROR: {config_path} not found. Cannot patch pyannote config.")
        return

    print("--- Patching pyannote/speaker-diarization-3.1 config.yaml ---")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # セグメンテーションモデルのパスをローカルパスに書き換える
    seg_model_path = MODELS_BASE_DIR / "pyannote-segmentation-3.0" / "pytorch_model.bin"
    config['pipeline']['params']['segmentation'] = str(seg_model_path)
    print(f"Patched segmentation path to: {seg_model_path}")

    # 埋め込みモデルのパスをローカルパスに書き換える
    emb_model_path = MODELS_BASE_DIR / "pyannote-wespeaker-voxceleb-resnet34-LM"
    config['pipeline']['params']['embedding'] = str(emb_model_path)
    print(f"Patched embedding path to: {emb_model_path}")

    # 変更後の設定を新しいファイルに保存
    new_config_path = diar_pipeline_dir / "local_config.yaml"
    with open(new_config_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"--- Patched config saved to {new_config_path} ---")

if __name__ == "__main__":
    hf_token = get_hf_token()
    if not hf_token:
        print("ERROR: Hugging Face token is not available. Set HUGGING_FACE_TOKEN or mount secret.")
    else:
        download_models(hf_token)
        patch_pyannote_config()