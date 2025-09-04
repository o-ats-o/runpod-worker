"""
ビルド時モデルベイキング:
 - Systran/faster-whisper-large-v3
 - pyannote/speaker-diarization-3.1 とその依存モデル
 - 実行時用の、ローカルパスに解決済みの diarization_config.yaml を生成
 - speechbrainモデルのhyperparams.yamlをオフライン用に書き換え
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download
import yaml
import sys
from hyperpyyaml import load_hyperpyyaml

# --- 環境変数とパス設定 ---
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not HF_TOKEN:
    raise SystemExit("ERROR: HF_TOKEN is required to bake models.")

hf_cache = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/app/hf_home/hub"))
models_root = Path("/app/models")

# Whisperモデルのバージョンをv3に更新
whisper_target = models_root / "whisper-large-v3"

# --- ディレクトリ作成 ---
models_root.mkdir(parents=True, exist_ok=True)
hf_cache.mkdir(parents=True, exist_ok=True)

# --- モデル格納先パス ---
pyannote_segmentation_target = models_root / "pyannote" / "segmentation-3.0"
pyannote_embedding_target    = models_root / "speechbrain" / "spkrec-ecapa-voxceleb"
config_output_path = Path("/app/diarization_config.yaml")

# --- ヘルパー関数 ---
def assert_exists(path: Path, desc: str):
    if not path.exists():
        print(f" Missing {desc}: {path}", file=sys.stderr)
        raise SystemExit(1)

def find_model_file(directory: Path) -> Path:
    possible_filenames = ["pytorch_model.bin", "embedding_model.ckpt", "model.ckpt", "model.safensors"]
    found_files = []
    for filename in possible_filenames:
        file_path = directory / filename
        if file_path.is_file():
            found_files.append(file_path)
    
    if not found_files:
        raise FileNotFoundError(f"No model checkpoint file found in {directory}. Searched for: {', '.join(possible_filenames)}")
    if len(found_files) > 1:
        raise ValueError(f"Multiple potential model files found in {directory}: {[str(f) for f in found_files]}. Cannot determine which one to use.")
    
    model_file = found_files[0]
    print(f"  - Found model checkpoint: {model_file}")
    return model_file

def copy_model_from_cache(repo_id: str, target_dir: Path, desc: str, revision: str = None):
    print(f" Downloading {desc} ({repo_id})...")
    try:
        snapshot_path_str = snapshot_download(repo_id=repo_id, use_auth_token=HF_TOKEN, cache_dir=str(hf_cache), revision=revision, local_files_only=False)
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

def rewrite_speechbrain_hyperparams(model_dir: Path, repo_id: str):
    """
    speechbrainモデルのhyperparams.yamlをテキストとして読み込み、
    `pretrained_path`の値をHuggingFaceリポジトリIDからコンテナ内の絶対パスに置換する。
    これにより、!new: タグなどのHyperPyYAMLの構造を完全に維持したまま、
    オフライン実行を可能にする。
    """
    hyperparams_path = model_dir / "hyperparams.yaml"
    if not hyperparams_path.is_file():
        print(f"Warning: hyperparams.yaml not found in {model_dir}. Skipping rewrite.")
        return

    print(f"Rewriting hyperparams.yaml for robust offline use in {model_dir}...")
    
    with open(hyperparams_path, 'r') as f:
        content = f.read()

    # HuggingFaceのリポジトリIDが指定されている行を、コンテナ内の絶対パスに置換する
    original_line = f"pretrained_path: {repo_id}"
    replacement_line = f"pretrained_path: {str(model_dir)}"
    
    if original_line in content:
        content = content.replace(original_line, replacement_line)
        print(f"  - Replaced '{original_line}' with '{replacement_line}'")
    else:
        print(f"  - WARNING: Did not find expected line '{original_line}' in hyperparams.yaml. The file might be structured differently.")
        # フォールバックとして、pretrained_pathで始まる行をすべて置換する
        import re
        content, count = re.subn(r"^pretrained_path:.*$", replacement_line, content, flags=re.MULTILINE)
        if count > 0:
            print(f"  - Fallback: Replaced line starting with 'pretrained_path:' with '{replacement_line}'")
        else:
            print("  - Fallback failed. No changes made.")

    with open(hyperparams_path, 'w') as f:
        f.write(content)
    
    print("hyperparams.yaml rewritten successfully.")

# --- Whisperモデル ---
# ダウンロードするモデルをv3に更新
copy_model_from_cache(repo_id="Systran/faster-whisper-large-v3", target_dir=whisper_target, desc="Whisper model")
assert_exists(whisper_target / "config.json", "whisper config.json")

# --- Pyannoteモデル ---
print("\nDownloading and copying pyannote models...")
segmentation_model_path = copy_model_from_cache(repo_id="pyannote/segmentation-3.0", target_dir=pyannote_segmentation_target, desc="Pyannote segmentation model")

embedding_repo_id = "speechbrain/spkrec-ecapa-voxceleb"
embedding_model_path = copy_model_from_cache(repo_id=embedding_repo_id, target_dir=pyannote_embedding_target, desc="Pyannote embedding model")

# --- SpeechBrainモデルの設定をオフライン用に修正 ---
rewrite_speechbrain_hyperparams(embedding_model_path, embedding_repo_id)

# --- diarization_config.yaml の生成 ---
print(f"\nGenerating configuration file at {config_output_path}...")
try:
    config_template_path = hf_hub_download(repo_id="pyannote/speaker-diarization-3.1", filename="config.yaml", use_auth_token=HF_TOKEN, cache_dir=str(hf_cache))
except Exception as e:
    print(f"Failed to download config.yaml for pyannote/speaker-diarization-3.1: {e}", file=sys.stderr)
    raise SystemExit(1)

with open(config_template_path, "r") as f:
    config_data = yaml.safe_load(f)

print("Rewriting model paths in config.yaml for offline use...")
# segmentationはチェックポイントファイルを直接指定
segmentation_checkpoint = find_model_file(segmentation_model_path)
config_data["pipeline"]["params"]["segmentation"] = str(segmentation_checkpoint)
# embeddingはspeechbrainモデルのディレクトリを指定
config_data["pipeline"]["params"]["embedding"] = str(embedding_model_path)

with open(config_output_path, "w") as f:
    yaml.dump(config_data, f, sort_keys=False, default_flow_style=False)

assert_exists(config_output_path, "Generated diarization config")
print("Configuration file generated successfully.")
print("\nAll models baked successfully into /app/models.")