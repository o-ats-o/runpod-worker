"""
ビルド時モデルベイク:
 - Systran/faster-whisper-large-v2 -> /app/models/whisper-large-v2
 - pyannote/speaker-diarization-3.1 の config.yaml
 - 依存: pyannote/segmentation-3.0, speechbrain/spkrec-ecapa-voxceleb
config.yaml 内の segmentation / embedding をローカル絶対パスに書き換え。
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import yaml

HF_TOKEN = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
)
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN (Hugging Face gated モデル用) が取得できません。BuildKit secret を渡してください。")

whisper_target = Path(os.environ.get("WHISPER_LOCAL_DIR", "/app/models/whisper-large-v2"))
pyannote_root = Path(os.environ.get("PYANNOTE_CACHE_DIR", "/app/models/pyannote"))
hf_cache = Path(os.environ.get("HUGGING_FACE_HUB_CACHE", "/app/models/hf_cache"))

for p in (whisper_target.parent, pyannote_root, hf_cache):
    p.mkdir(parents=True, exist_ok=True)

# ---------------- Whisper (faster-whisper large-v2) ----------------
print("[Bake] Downloading faster-whisper large-v2 ...")
whisper_repo = "Systran/faster-whisper-large-v2"
repo_path = snapshot_download(
    repo_id=whisper_repo,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
if whisper_target.exists():
    shutil.rmtree(whisper_target)
shutil.copytree(repo_path, whisper_target)
print(f"[Bake] Whisper model stored at: {whisper_target}")

# ---------------- pyannote pipeline & dependencies ----------------
print("[Bake] Downloading pyannote + dependencies ...")
PIPELINE = "pyannote/speaker-diarization-3.1"
SEGMENTATION = "pyannote/segmentation-3.0"
EMBEDDING = "speechbrain/spkrec-ecapa-voxceleb"

pipeline_config_path = hf_hub_download(
    repo_id=PIPELINE,
    filename="config.yaml",
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
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

pipeline_dir = pyannote_root / "pipelines" / "speaker-diarization-3.1"
if pipeline_dir.exists():
    shutil.rmtree(pipeline_dir)
pipeline_dir.mkdir(parents=True, exist_ok=True)

target_config = pipeline_dir / "config.yaml"
shutil.copy2(pipeline_config_path, target_config)

with open(target_config, "r") as f:
    config_data = yaml.safe_load(f)

params = config_data.setdefault("pipeline", {}).setdefault("params", {})
params["segmentation"] = seg_dir
params["embedding"] = emb_dir

with open(target_config, "w") as f:
    yaml.dump(config_data, f, sort_keys=False)

print(f"[Bake] pyannote pipeline config written at: {target_config}")
print("[Bake] All models baked successfully.")