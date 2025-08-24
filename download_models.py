"""
ビルド時モデルベイク:
 - Systran/faster-whisper-large-v2
 - pyannote/speaker-diarization-3.1 とその依存モデル
 - 実行時用の diarization_config.yaml を生成
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml
import sys

# --- 環境変数とパス設定 ---
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required to bake models.")

hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/app/hf_home/hub"))
whisper_target = Path(os.environ.get("WHISPER_LOCAL_DIR", "/app/models/whisper-large-v2"))
pyannote_root = Path(os.environ.get("PYANNOTE_CACHE_DIR", "/app/models/pyannote"))
config_output_path = Path("/app/diarization_config.yaml")

# --- ディレクトリ作成 ---
for p in (whisper_target.parent, pyannote_root, hf_cache):
    p.mkdir(parents=True, exist_ok=True)

# --- ヘルパー関数 ---
def assert_exists(path: Path, desc: str):
    if not path.exists():
        print(f" Missing {desc}: {path}", file=sys.stderr)
        raise SystemExit(1)

# --- Whisperモデルのダウンロード ---
print(" Downloading faster-whisper large-v2...")
whisper_repo = "Systran/faster-whisper-large-v2"
whisper_snapshot_path = snapshot_download(
    repo_id=whisper_repo,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)

print(f" Copying whisper model from {whisper_snapshot_path} to {whisper_target}...")
if whisper_target.exists():
    shutil.rmtree(whisper_target)
shutil.copytree(whisper_snapshot_path, whisper_target)
assert_exists(whisper_target / "config.json", "whisper config.json")
print(" Whisper model copied successfully.")


# --- Pyannoteモデルのダウンロード ---
print(" Downloading pyannote models...")
PIPELINE_REPO = "pyannote/speaker-diarization-3.1"
SEGMENTATION_REPO = "pyannote/segmentation-3.0"
EMBEDDING_REPO = "speechbrain/spkrec-ecapa-voxceleb"

# パイプラインとその依存モデルをダウンロード
# snapshot_downloadはダウンロードされたディレクトリの絶対パスを返す
pipeline_snapshot_path = snapshot_download(repo_id=PIPELINE_REPO, use_auth_token=HF_TOKEN, cache_dir=str(hf_cache))
segmentation_model_path = snapshot_download(repo_id=SEGMENTATION_REPO, use_auth_token=HF_TOKEN, cache_dir=str(hf_cache))
embedding_model_path = snapshot_download(repo_id=EMBEDDING_REPO, use_auth_token=HF_TOKEN, cache_dir=str(hf_cache))

print(f" Pipeline snapshot path: {pipeline_snapshot_path}")
print(f" Segmentation model path: {segmentation_model_path}")
print(f" Embedding model path: {embedding_model_path}")

# --- diarization_config.yaml の生成 ---
print(f" Generating configuration file at {config_output_path}...")

config_data = {
    "pipeline": {
        "name": "pyannote.audio.pipelines.SpeakerDiarization",
        "params": {
            # ダウンロードしたモデルの実際のパスを指定
            "segmentation": segmentation_model_path,
            "embedding": embedding_model_path,
            "clustering": "AgglomerativeClustering",
            "embedding_exclude_overlap": True,
        }
    },
    "params": {
        "segmentation": {
            "min_duration_off": 0.0,
        },
        "clustering": {
            "method": "centroid",
            "min_cluster_size": 15,
        }
    }
}

with open(config_output_path, "w") as f:
    yaml.dump(config_data, f, sort_keys=False)

assert_exists(config_output_path, "Generated diarization config")
print(" Configuration file generated successfully.")
print(" All models baked successfully.")