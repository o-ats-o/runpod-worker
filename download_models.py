"""
ビルド時モデルベイキング:
 - Systran/faster-whisper-large-v2
 - pyannote/speaker-diarization-3.1 とその依存モデル
 - 実行時用の、ローカルパスに解決済みの diarization_config.yaml を生成
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import yaml
import sys

# --- 環境変数とパス設定 ---
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required to bake models.")

# hf_cacheはダウンロード時のみに使用
hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/app/hf_home/hub"))
# 全てのモデルをこの単一のディレクトリに集約
models_root = Path("/app/models")
whisper_target = models_root / "whisper-large-v2"

# --- ディレクトリ作成 ---
models_root.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)

# --- モデル格納先パス (コンテナ内での最終的な絶対パス) ---
pyannote_segmentation_target = models_root / "pyannote" / "segmentation-3.0"
pyannote_embedding_target    = models_root / "speechbrain" / "spkrec-ecapa-voxceleb"
config_output_path = Path("/app/diarization_config.yaml")

# --- ヘルパー関数 ---
def assert_exists(path: Path, desc: str):
    if not path.exists():
        print(f" Missing {desc}: {path}", file=sys.stderr)
        raise SystemExit(1)

def copy_model_from_cache(repo_id: str, target_dir: Path, desc: str, revision: str = None):
    print(f" Downloading {desc} ({repo_id})...")
    try:
        snapshot_path_str = snapshot_download(
            repo_id=repo_id,
            use_auth_token=HF_TOKEN,
            cache_dir=str(hf_cache),
            revision=revision,
            local_files_only=False,
        ) 
    except Exception as e:
        print(f"Failed to download {repo_id}: {e}", file=sys.stderr)
        raise SystemExit(1)
        
    snapshot_path = Path(snapshot_path_str)
    print(f" Copying {desc} from {snapshot_path} to {target_dir}...")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_path, target_dir)
    print(f" {desc} copied successfully.")
    return target_dir

# --- Whisperモデルのダウンロードとコピー ---
copy_model_from_cache(
    repo_id="Systran/faster-whisper-large-v2",
    target_dir=whisper_target,
    desc="Whisper model"
)
assert_exists(whisper_target / "config.json", "whisper config.json")

# --- Pyannoteモデルのダウンロードとコピー ---
print("\nDownloading and copying pyannote models...")
segmentation_model_path = copy_model_from_cache(
    repo_id="pyannote/segmentation-3.0",
    target_dir=pyannote_segmentation_target,
    desc="Pyannote segmentation model"
)
embedding_model_path = copy_model_from_cache(
    repo_id="speechbrain/spkrec-ecapa-voxceleb",
    target_dir=pyannote_embedding_target,
    desc="Pyannote embedding model"
)

# --- diarization_config.yaml の生成 ---
print(f"\nGenerating configuration file at {config_output_path}...")
print("Downloading official pyannote/speaker-diarization-3.1 config.yaml...")
try:
    config_template_path = hf_hub_download(
        repo_id="pyannote/speaker-diarization-3.1",
        filename="config.yaml",
        use_auth_token=HF_TOKEN,
        cache_dir=str(hf_cache),
    )
except Exception as e:
    print(f"Failed to download config.yaml for pyannote/speaker-diarization-3.1: {e}", file=sys.stderr)
    raise SystemExit(1)

with open(config_template_path, "r") as f:
    config_data = yaml.safe_load(f)

print("Rewriting model paths in config.yaml for offline use...")
# segmentation は pytorch_model.bin ファイルを直接指す
config_data["pipeline"]["params"]["segmentation"] = str(segmentation_model_path / "pytorch_model.bin")
# embedding は公開モデルIDのまま
config_data["pipeline"]["params"]["embedding"] = "speechbrain/spkrec-ecapa-voxceleb"

with open(config_output_path, "w") as f:
    yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

assert_exists(config_output_path, "Generated diarization config")
print("Configuration file generated successfully.")
print("\nAll models baked successfully into /app/models.")
