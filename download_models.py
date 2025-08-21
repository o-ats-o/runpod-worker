"""
ビルド時モデルベイク:
 - Systran/faster-whisper-large-v2 (snapshot -> symlink WHISPER_LOCAL_DIR)
 - pyannote/speaker-diarization-3.1 (丸ごと snapshot)
 - 依存: pyannote/segmentation-3.0, speechbrain/spkrec-ecapa-voxceleb
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required to bake gated models.")

whisper_target = Path(os.environ.get("WHISPER_LOCAL_DIR", "/app/models/whisper-large-v2"))
pyannote_root  = Path(os.environ.get("PYANNOTE_CACHE_DIR", "/app/models/pyannote"))
hf_cache       = Path(os.environ.get("HUGGING_FACE_HUB_CACHE", "/app/models/hf_cache"))

for p in (whisper_target.parent, pyannote_root, hf_cache):
    p.mkdir(parents=True, exist_ok=True)

# Whisper snapshot
print("[Bake] Downloading faster-whisper large-v2 ...")
whisper_repo = "Systran/faster-whisper-large-v2"
whisper_snapshot = snapshot_download(
    repo_id=whisper_repo,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
# Symlink to avoid duplication
if whisper_target.exists() or whisper_target.is_symlink():
    if whisper_target.is_dir() and not whisper_target.is_symlink():
        shutil.rmtree(whisper_target)
    else:
        whisper_target.unlink()
whisper_target.symlink_to(whisper_snapshot)
print(f"[Bake] Whisper symlink: {whisper_target} -> {whisper_snapshot}")

# Pyannote pipeline snapshot (full)
print("[Bake] Downloading pyannote speaker-diarization-3.1 ...")
PIPELINE = "pyannote/speaker-diarization-3.1"
pipeline_snapshot = snapshot_download(
    repo_id=PIPELINE,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)

# Dependencies (segmentation / embedding)
print("[Bake] Downloading pyannote segmentation + speechbrain embedding ...")
SEGMENTATION = "pyannote/segmentation-3.0"
EMBEDDING = "speechbrain/spkrec-ecapa-voxceleb"
seg_dir = snapshot_download(
    repo_id=SEGMENTATION,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
emb_dir = snapshot_download(
    repo_id=EMBEDDING,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)

# Create local pipeline config directory (mirrors earlier approach, but now we can just rewrite a copy)
pipeline_dir = pyannote_root / "pipelines" / "speaker-diarization-3.1"
if pipeline_dir.exists():
    shutil.rmtree(pipeline_dir)
pipeline_dir.mkdir(parents=True, exist_ok=True)

# Copy original config from snapshot
orig_config = Path(pipeline_snapshot) / "config.yaml"
if not orig_config.exists():
    raise SystemExit("ERROR: pipeline snapshot missing config.yaml")
local_config = pipeline_dir / "config.yaml"
shutil.copy2(orig_config, local_config)

# Rewrite segmentation & embedding to absolute snapshot paths
with open(local_config, "r") as f:
    cfg = yaml.safe_load(f) or {}
params = cfg.setdefault("pipeline", {}).setdefault("params", {})
params["segmentation"] = seg_dir
params["embedding"] = emb_dir
with open(local_config, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)

print(f"[Bake] Rewritten pipeline config: {local_config}")
print("[Bake] All models baked successfully.")