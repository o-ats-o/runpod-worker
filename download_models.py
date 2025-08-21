"""
ビルド時モデルベイク:
 - Systran/faster-whisper-large-v2 (snapshot -> symlink WHISPER_LOCAL_DIR)
 - pyannote/speaker-diarization-3.1 (snapshot -> symlink PYANNOTE_CACHE_DIR/pipeline_snapshot)
 - 依存: pyannote/segmentation-3.0, speechbrain/spkrec-ecapa-voxceleb
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import yaml
import sys

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required to bake gated models.")

hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
whisper_target = Path(os.environ.get("WHISPER_LOCAL_DIR", "/app/models/whisper-large-v2"))
pyannote_root = Path(os.environ.get("PYANNOTE_CACHE_DIR", "/app/models/pyannote"))
pyannote_symlink = pyannote_root / "pipeline_snapshot"

for p in (whisper_target.parent, pyannote_root, hf_cache):
    p.mkdir(parents=True, exist_ok=True)

def ensure_dir_clean_link(link_path: Path, target_path: Path):
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target_path)

def assert_exists(path: Path, desc: str):
    if not path.exists():
        print(f"[Bake][ERROR] Missing {desc}: {path}", file=sys.stderr)
        raise SystemExit(1)

print("[Bake] Downloading faster-whisper large-v2 ...")
whisper_repo = "Systran/faster-whisper-large-v2"
whisper_snapshot = snapshot_download(
    repo_id=whisper_repo,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
ensure_dir_clean_link(whisper_target, Path(whisper_snapshot))
assert_exists(whisper_target / "config.json", "whisper config.json")
print(f"[Bake] Whisper symlink: {whisper_target} -> {whisper_snapshot}")

print("[Bake] Downloading pyannote speaker-diarization-3.1 ...")
PIPELINE = "pyannote/speaker-diarization-3.1"
pipeline_snapshot = snapshot_download(
    repo_id=PIPELINE,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
ensure_dir_clean_link(pyannote_symlink, Path(pipeline_snapshot))
snapshot_config = Path(pipeline_snapshot) / "config.yaml"
assert_exists(snapshot_config, "pipeline config.yaml")

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
assert_exists(Path(seg_dir) / "config.yaml", "segmentation config")
# emb_dir may not have config.yaml; skip strict check.

# Rewrite snapshot config
with open(snapshot_config, "r") as f:
    cfg = yaml.safe_load(f) or {}
params = cfg.setdefault("pipeline", {}).setdefault("params", {})
params["segmentation"] = seg_dir
params["embedding"] = emb_dir
with open(snapshot_config, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)

print(f"[Bake] Rewritten snapshot config: {snapshot_config}")
print("[Bake] All models baked successfully.")