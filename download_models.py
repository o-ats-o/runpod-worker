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

# hf_cacheはダウンロード時のみに使用
hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/app/hf_home/hub"))
# 全てのモデルをこの単一のディレクトリに集約
models_root = Path("/app/models")
whisper_target = models_root / "whisper-large-v2"
pyannote_segmentation_target = models_root / "pyannote-segmentation-3.0"
pyannote_embedding_target = models_root / "pyannote-embedding-ecapa"
config_output_path = Path("/app/diarization_config.yaml")

# --- ディレクトリ作成 ---
models_root.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)


# --- ヘルパー関数 ---
def assert_exists(path: Path, desc: str):
    if not path.exists():
        print(f" Missing {desc}: {path}", file=sys.stderr)
        raise SystemExit(1)

def copy_model_from_cache(repo_id: str, target_dir: Path, desc: str):
    """指定されたrepoをダウンロードし、キャッシュからターゲットディレクトリにコピーする"""
    print(f" Downloading {desc} ({repo_id})...")
    snapshot_path_str = snapshot_download(
        repo_id=repo_id,
        use_auth_token=HF_TOKEN,
        cache_dir=str(hf_cache),
        # allow_patternsを使用して不要なファイルをダウンロードしないようにする（オプション）
        # 例: allow_patterns=["*.bin", "*.json", "*.txt"]
    )
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
print(" Downloading and copying pyannote models...")
# パイプライン自体のファイルは不要なため、依存モデルのみを直接取得・コピー
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
# ここが重要：キャッシュの絶対パスではなく、イメージ内の固定パスを使用
print(f" Generating configuration file at {config_output_path}...")

config_data = {
    "pipeline": {
        "name": "pyannote.audio.pipelines.SpeakerDiarization",
        "params": {
            # 最終イメージ内の整理されたパスを指定
            "segmentation": str(segmentation_model_path),
            "embedding": str(embedding_model_path),
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
print(" All models baked successfully into /app/models.")