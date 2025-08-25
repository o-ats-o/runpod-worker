"""
ビルド時モデルベイク:
 - Systran/faster-whisper-large-v2
 - pyannote/segmentation-3.0
 - speechbrain/spkrec-ecapa-voxceleb
 - diarization_config.yaml をローカルパス指定で生成
"""
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml
import sys

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required to bake models.")

hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/app/hf_home/hub"))
models_root = Path("/app/models")
whisper_target = models_root / "whisper-large-v2"
(models_root / "pyannote").mkdir(parents=True, exist_ok=True)
(models_root / "speechbrain").mkdir(parents=True, exist_ok=True)
models_root.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)

pyannote_segmentation_target = models_root / "pyannote" / "segmentation-3.0"
pyannote_embedding_target   = models_root / "speechbrain" / "spkrec-ecapa-voxceleb"
config_output_path = Path("/app/diarization_config.yaml")

def assert_exists(path: Path, desc: str):
    if not path.exists():
        print(f" Missing {desc}: {path}", file=sys.stderr)
        raise SystemExit(1)

def copy_model_from_cache(repo_id: str, target_dir: Path, desc: str):
    print(f" Downloading {desc} ({repo_id})...")
    snapshot_path_str = snapshot_download(
        repo_id=repo_id,
        use_auth_token=HF_TOKEN,
        cache_dir=str(hf_cache),
    )
    snapshot_path = Path(snapshot_path_str)
    print(f" Copying {desc} from {snapshot_path} to {target_dir}...")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_path, target_dir)
    print(f" {desc} copied successfully.")
    return target_dir

# Whisper
copy_model_from_cache("Systran/faster-whisper-large-v2", whisper_target, "Whisper model")
assert_exists(whisper_target / "config.json", "whisper config.json")

# Pyannote
segmentation_model_path = copy_model_from_cache("pyannote/segmentation-3.0", pyannote_segmentation_target, "Pyannote segmentation")
embedding_model_path = copy_model_from_cache("speechbrain/spkrec-ecapa-voxceleb", pyannote_embedding_target, "Pyannote embedding")

# diarization_config.yaml をローカルパスで生成
config_data = {
    "pipeline": {
        "name": "pyannote.audio.pipelines.SpeakerDiarization",
        "params": {
            "segmentation": str(segmentation_model_path),
            "embedding": str(embedding_model_path),
            "clustering": "AgglomerativeClustering",
            "embedding_exclude_overlap": True,
        }
    },
    "params": {
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {"method": "centroid", "min_cluster_size": 15}
    }
}
with open(config_output_path, "w") as f:
    yaml.dump(config_data, f, sort_keys=False)

assert_exists(config_output_path, "Generated diarization config")
print(" Configuration file generated successfully.")