import os
import base64
import tempfile
import warnings
import torch
import torchaudio
from typing import List, Dict, Any, Optional
import runpod
from pathlib import Path

# --- 警告フィルタ ---
warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
warnings.filterwarnings("ignore", module="pyannote.audio.models.blocks.pooling")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 定数と環境変数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_LANGUAGE = os.environ.get("DEFAULT_TRANSCRIBE_LANG", "ja")
HF_TOKEN = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
WHISPER_LOCAL_DIR = os.environ.get("WHISPER_LOCAL_DIR", "/app/models/whisper-large-v2")
# ---------------------------------------------
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "large-v2")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

# ビルド時に生成される設定ファイルのパス
DIARIZATION_CONFIG_PATH = "/app/diarization_config.yaml"

# PyTorchパフォーマンス設定
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- グローバルモデル変数 ---
_whisper_model = None
_diarization_pipeline = None

# ... (format_timestamp, _segment_durationなどのヘルパー関数は変更なし)...
def format_timestamp(seconds: float) -> str:
    ms = round(seconds * 1000)
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _segment_duration(seg: Dict[str, Any]) -> float:
    return float(seg["end"]) - float(seg["start"])

def _merge_adjacent_same_speaker(segments: List[Dict[str, Any]], max_gap: float = 0.1) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for seg in segments[1:]:
        last = merged[-1]
        if seg["speaker"] == last["speaker"] and float(seg["start"]) - float(last["end"]) <= max_gap:
            last["end"] = seg["end"]
            if seg.get("text"):
                last["text"] = (last.get("text", "") + " " + seg["text"]).strip()
        else:
            merged.append(dict(seg))
    return merged

def _compute_overlap_ratio(s0: float, s1: float, t0: float, t1: float) -> float:
    length = max(0.0, s1 - s0)
    if length <= 0: return 0.0
    ov = min(s1, t1) - max(s0, t0)
    if ov <= 0: return 0.0
    return ov / length
def best_speaker_for_interval(start: float, end: float, diarization_result, threshold: float) -> str:
    best, best_ratio = "UNKNOWN_SPEAKER", 0.0
    for turn, _, spk in diarization_result.itertracks(yield_label=True):
        ratio = _compute_overlap_ratio(start, end, turn.start, turn.end)
        if ratio > best_ratio: best_ratio, best = ratio, spk
    return best if best_ratio >= threshold else "UNKNOWN_SPEAKER"
def max_overlap_speaker(start: float, end: float, diarization_result, prev: Optional[str]) -> str:
    best, best_ov = "UNKNOWN_SPEAKER", 0.0
    for turn, _, spk in diarization_result.itertracks(yield_label=True):
        ov = min(end, turn.end) - max(start, turn.start)
        if ov > best_ov: best_ov, best = ov, spk
    if best == "UNKNOWN_SPEAKER" and prev and prev!= "UNKNOWN_SPEAKER": return prev
    return best
def determine_speaker(segment, diarization_result, prev: str, overlap_ratio_threshold: float) -> str:
    words = getattr(segment, "words", None)
    if not words: return max_overlap_speaker(segment.start, segment.end, diarization_result, prev)
    votes = {}
    for w in words:
        ws, we = getattr(w, "start", None), getattr(w, "end", None)
        if ws is None or we is None or we <= ws: continue
        spk = best_speaker_for_interval(ws, we, diarization_result, overlap_ratio_threshold)
        if spk!= "UNKNOWN_SPEAKER": votes[spk] = votes.get(spk, 0.0) + (we - ws)
    if not votes: return prev if prev and prev!= "UNKNOWN_SPEAKER" else max_overlap_speaker(segment.start, segment.end, diarization_result, prev)
    return max(list(votes.items()), key=lambda kv: kv[1])[0]
def smooth_labels(segments: List[Dict[str, Any]], merge_short_threshold: float, hold: float) -> List[Dict[str, Any]]:
    if not segments:
        return []
    segs = [dict(s) for s in segments]
    if hold > 0:
        prev = segs[0]["speaker"]
        for i in range(1, len(segs)):
            cur = segs[i]
            if cur["speaker"] != prev and _segment_duration(cur) < hold and prev != "UNKNOWN_SPEAKER":
                cur["speaker"] = prev
            else:
                prev = cur["speaker"]
    if merge_short_threshold > 0 and len(segs) >= 3:
        for i in range(1, len(segs)-1):
            left, cur, right = segs[i-1], segs[i], segs[i+1]
            if (cur["speaker"] != left["speaker"] and cur["speaker"] != right["speaker"] and
                _segment_duration(cur) < merge_short_threshold and
                left["speaker"] == right["speaker"] and left["speaker"] != "UNKNOWN_SPEAKER"):
                cur["speaker"] = left["speaker"]
    segs = _merge_adjacent_same_speaker(segs)
    return segs

# --- モデル読み込みロジック ---

def load_diarization_pipeline():
    """
    ビルド時に生成された設定ファイルからローカルモデルをロード。
    Hugging Face Hub は使わない。
    """
    import yaml
    from pyannote.audio.pipelines import SpeakerDiarization
    from pyannote.audio.core.model import Model

    print(f"[Init] Load Pyannote pipeline (offline) from {DIARIZATION_CONFIG_PATH}")
    cfg_path = Path(DIARIZATION_CONFIG_PATH)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Diarization config not found at {DIARIZATION_CONFIG_PATH}.")

    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)

    seg_path = config["pipeline"]["params"]["segmentation"]
    emb_path = config["pipeline"]["params"]["embedding"]

    # モデルをローカルパスからロード
    seg_model = Model.from_pretrained(seg_path)
    emb_model = Model.from_pretrained(emb_path)

    pipeline = SpeakerDiarization(
        segmentation=seg_model,
        embedding=emb_model,
        clustering=config["pipeline"]["params"]["clustering"],
        embedding_exclude_overlap=config["pipeline"]["params"].get("embedding_exclude_overlap", True),
    )
    pipeline.instantiate(config.get("params", {}))
    return pipeline

def ensure_models_loaded():
    """
    WhisperとDiarizationモデルがロードされていることを確認する。
    """
    global _whisper_model, _diarization_pipeline
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        # ローカルディレクトリの存在確認のみで十分。オンラインフォールバックは不要。
        if os.path.isdir(WHISPER_LOCAL_DIR):
            print(f"[Init] Load Whisper local: {WHISPER_LOCAL_DIR}")
            _whisper_model = WhisperModel(WHISPER_LOCAL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
        else:
            # このエラーはビルドが失敗していることを示す
            raise FileNotFoundError(f"Whisper model directory not found at {WHISPER_LOCAL_DIR}. The Docker image may be built incorrectly.")
    
    if _diarization_pipeline is None:
        _diarization_pipeline = load_diarization_pipeline()
        _diarization_pipeline.to(torch.device(DEVICE))

# --- RunPodハンドラ ---
def handler(job):
    try:
        ensure_models_loaded()
        params = job.get("input", {}) if isinstance(job, dict) else {}
        b64 = params.get("audio_base64")
        if not b64:
            return {"error": "audio_base64 がありません。"}

        language = params.get("language", DEFAULT_LANGUAGE)
        overlap_ratio_threshold = float(params.get("overlap_ratio_threshold", 0.3))
        merge_short_threshold = float(params.get("merge_short_segment_threshold", 0.5))
        speaker_hold_time = float(params.get("speaker_hold_time", 0.8))

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(base64.b64decode(b64))
            tmp_path = tmp.name

        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
        except Exception as e:
            os.unlink(tmp_path)
            return {"error": f"音声読み込み失敗: {e}"}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        print("[Job] Diarization...")
        diarization_result = _diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

        print("[Job] Transcription...")
        segments_iter, info = _whisper_model.transcribe(
            waveform.squeeze(0).numpy(),
            beam_size=5,
            language=language if language and language.lower()!= "none" else None,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=250),
            word_timestamps=True,
        )
        segments = list(segments_iter)

        print("[Job] Combine + speaker assignment...")
        combined = []
        prev = "UNKNOWN_SPEAKER"
        for seg in segments:
            spk = determine_speaker(seg, diarization_result, prev, overlap_ratio_threshold)
            combined.append({
                "speaker": spk,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip()
            })
            prev = spk

        smoothed = smooth_labels(combined, merge_short_threshold, speaker_hold_time)
        final = [{
            "speaker": s["speaker"],
            "start_time": format_timestamp(s["start"]),
            "end_time": format_timestamp(s["end"]),
            "text": s.get("text","")
        } for s in smoothed]

        return {
            "language": getattr(info, "language", language),
            "segments": final
        }
    except Exception as e:
        import traceback
        print(f"Error during job execution: {e}")
        traceback.print_exc()
        return {"error": f"処理中に例外が発生しました: {e}"}


# --- サーバー起動 ---
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})