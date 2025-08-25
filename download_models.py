"""
ビルド時モデルベイク（オンライン）:
 - pyannote/speaker-diarization, pyannote/segmentation, speechbrain/spkrec-ecapa-voxceleb
 - ローカル用 config.yml (Speech-Diarization のオフライン設定) を生成
"""
import os, shutil, sys, yaml
from pathlib import Path
from huggingface_hub import snapshot_download

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required.")

hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/app/hf_home/hub"))
models_root = Path("/app/models")
models_root.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)

seg_dir = models_root / "segmentation"
emb_dir = models_root / "spkrec-ecapa-voxceleb"
config_out = Path("/app/diarization_config.yml")

def dl(repo_id, target, desc):
    path = Path(snapshot_download(repo_id=repo_id, use_auth_token=HF_TOKEN, cache_dir=str(hf_cache)))
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(path, target)
    return target

seg_model = dl("pyannote/segmentation-3.0", seg_dir, "segmentation")
emb_model = dl("speechbrain/spkrec-ecapa-voxceleb", emb_dir, "embedding")

config = {
    "pipeline": {
        "name": "pyannote.audio.pipelines.SpeakerDiarization",
        "params": {
            "segmentation": str(seg_model),
            "embedding": str(emb_model),
            "clustering": "AgglomerativeClustering",
            "embedding_exclude_overlap": True
        }
    },
    "params": {
        "segmentation": {"min_duration_off": 0.0},
        "clustering": {"method": "centroid", "min_cluster_size": 15}
    }
}

with open(config_out, "w") as f:
    yaml.dump(config, f, sort_keys=False)
print(f"Generated offline config at {config_out}")