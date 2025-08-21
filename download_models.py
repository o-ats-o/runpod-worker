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

def ensure_ref(repo_id: str, snapshot_path: Path):
    """
    repo_id: 'namespace/name'
    snapshot_path: .../snapshots/<commit_hash>
    refs/main が無ければ作成する
    """
    commit_hash = snapshot_path.name
    if len(commit_hash) != 40 or not all(c in "0123456789abcdef" for c in commit_hash.lower()):
        print(f"[Bake][WARN] Snapshot tail '{commit_hash}' does not look like a commit SHA.")
    repo_cache_dir = hf_cache / f"models--{repo_id.replace('/','--')}"
    refs_dir = repo_cache_dir / "refs"
    refs_dir.mkdir(exist_ok=True)
    main_ref = refs_dir / "main"
    if not main_ref.exists():
        main_ref.write_text(commit_hash)
        print(f"[Bake] Created refs/main for {repo_id} -> {commit_hash}")
    else:
        existing = main_ref.read_text().strip()
        if existing != commit_hash:
            # 競合する場合は最新 snapshot を優先して上書き
            main_ref.write_text(commit_hash)
            print(f"[Bake] Updated refs/main ({existing} -> {commit_hash})")
        else:
            print(f"[Bake] refs/main already OK ({commit_hash})")
    # 最終チェック
    if not main_ref.exists():
        raise SystemExit(f"[Bake][ERROR] Failed to ensure refs/main for {repo_id}")
    return main_ref

print("[Bake] Downloading faster-whisper large-v2 ...")
whisper_repo = "Systran/faster-whisper-large-v2"
whisper_snapshot = snapshot_download(
    repo_id=whisper_repo,
    use_auth_token=HF_TOKEN,
    cache_dir=str(hf_cache),
)
ensure_dir_clean_link(whisper_target, Path(whisper_snapshot))
assert_exists(whisper_target / "config.json", "whisper config.json")
ensure_ref(whisper_repo, Path(whisper_snapshot))
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
ensure_ref(PIPELINE, Path(pipeline_snapshot))

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
ensure_ref(SEGMENTATION, Path(seg_dir))
ensure_ref(EMBEDDING, Path(emb_dir))

# Rewrite snapshot config (pipeline)
with open(snapshot_config, "r") as f:
    cfg = yaml.safe_load(f) or {}
params = cfg.setdefault("pipeline", {}).setdefault("params", {})
params["segmentation"] = seg_dir
params["embedding"] = emb_dir
with open(snapshot_config, "w") as f:
    yaml.dump(cfg, f, sort_keys=False)

print(f"[Bake] Rewritten snapshot config: {snapshot_config}")

# 最終表示（パス + refs 確認）
for rid in (whisper_repo, PIPELINE, SEGMENTATION, EMBEDDING):
    repo_cache_dir = hf_cache / f"models--{rid.replace('/','--')}"
    ref_file = repo_cache_dir / "refs" / "main"
    print(f"[Bake] refs check {rid}: {ref_file.exists()} ({ref_file.read_text().strip() if ref_file.exists() else 'N/A'})")

print("[Bake] All models baked successfully.")