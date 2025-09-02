import os
import tempfile
import warnings
import torch
import torchaudio
from typing import List, Dict, Any, Optional
import runpod
from pathlib import Path
import logging
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

# --- デバッグログ設定 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# --------------------------

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
WHISPER_MODEL_NAME = os.environ.get("WHISPER_MODEL_NAME", "large-v2")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
DIARIZATION_CONFIG_PATH = "/app/diarization_config.yaml"

# PyTorchパフォーマンス設定
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- グローバルモデル変数 ---
_whisper_model = None
_diarization_pipeline = None
_s3_client = None

# --- R2クライアント初期化関数 ---
def init_s3_client():
    """環境変数からR2の認証情報を読み込み、S3クライアントを初期化する"""
    global _s3_client
    try:
        account_id = os.environ["R2_ACCOUNT_ID"]
        access_key_id = os.environ["R2_ACCESS_KEY_ID"]
        secret_access_key = os.environ["R2_SECRET_ACCESS_KEY"]
        
        _s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            config=Config(signature_version='s3v4') # R2との互換性のために必要
        )
        logging.info("Cloudflare R2 client initialized successfully.")
    except KeyError as e:
        logging.error(f"R2の環境変数が設定されていません: {e}")
        _s3_client = None
    except Exception as e:
        logging.error(f"R2クライアントの初期化に失敗しました: {e}")
        _s3_client = None
        
# --- モデル読み込みロジック ---
def ensure_models_loaded():
    """WhisperとDiarizationモデルがロードされていることを確認する"""
    global _whisper_model, _diarization_pipeline
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logging.info("--- Loading Whisper model... ---")
        if os.path.isdir(WHISPER_LOCAL_DIR):
            _whisper_model = WhisperModel(WHISPER_LOCAL_DIR, device=DEVICE, compute_type=COMPUTE_TYPE)
            logging.info("--- Whisper model loaded successfully. ---")
        else:
            raise FileNotFoundError(f"Whisper model directory not found at {WHISPER_LOCAL_DIR}.")
    
    if _diarization_pipeline is None:
        from pyannote.audio import Pipeline
        logging.info(f"--- 1. Starting to load Pyannote pipeline from config: {DIARIZATION_CONFIG_PATH} ---")
        cfg_path = Path(DIARIZATION_CONFIG_PATH)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Diarization config not found at {DIARIZATION_CONFIG_PATH}.")
        _diarization_pipeline = Pipeline.from_pretrained(cfg_path, use_auth_token=False)
        logging.info("--- 2. Pyannote pipeline object created successfully. ---")
        _diarization_pipeline.to(torch.device(DEVICE))
        logging.info(f"--- 3. Pyannote pipeline moved to device: {DEVICE} ---")


# --- フォーマット関数 ---
def format_timestamp(seconds: float, always_include_hours: bool = False) -> str:
    ms = round(seconds * 1000)
    h = ms // 3_600_000
    ms %= 3_600_000
    m = ms // 60_000
    ms %= 60_000
    s = ms // 1000
    ms %= 1000
    if always_include_hours or h > 0:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    else:
        return f"{m:02d}:{s:02d},{ms:03d}"

def preprocess_audio_to_mono_16k(waveform: torch.Tensor, sample_rate: int, target_rate: int = 16000):
    """
    入力の `waveform` (C x T または T) をモノラル・target_rate(既定16kHz) に変換して返す。
    戻り値は (T,), sample_rate。
    """
    # チャンネル次元を確実に持たせる
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    # 複数チャネルは平均でモノラル化
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # サンプリングレート変換
    if sample_rate != target_rate:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_rate)
        sample_rate = target_rate
    return waveform.squeeze(0), sample_rate

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
    if not words:
        if prev and prev != "UNKNOWN_SPEAKER":
            return prev
        return max_overlap_speaker(segment.start, segment.end, diarization_result, prev)
    votes = {}
    for w in words:
        ws, we = getattr(w, "start", None), getattr(w, "end", None)
        if ws is None or we is None or we <= ws:
            continue
        spk = best_speaker_for_interval(ws, we, diarization_result, overlap_ratio_threshold)
        if spk != "UNKNOWN_SPEAKER":
            votes[spk] = votes.get(spk, 0.0) + (we - ws)
    if not votes:
        return prev if prev and prev != "UNKNOWN_SPEAKER" else max_overlap_speaker(segment.start, segment.end, diarization_result, prev)
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
        for i in range(1, len(segs) - 1):
            left, cur, right = segs[i - 1], segs[i], segs[i + 1]
            if (cur["speaker"] != left["speaker"] and cur["speaker"] != right["speaker"] and
                _segment_duration(cur) < merge_short_threshold):
                if left["speaker"] == right["speaker"] and left["speaker"] != "UNKNOWN_SPEAKER":
                    cur["speaker"] = left["speaker"]
                else:
                    left_dur = _segment_duration(left) if left["speaker"] != "UNKNOWN_SPEAKER" else -1.0
                    right_dur = _segment_duration(right) if right["speaker"] != "UNKNOWN_SPEAKER" else -1.0
                    if left_dur >= right_dur and left_dur >= 0:
                        cur["speaker"] = left["speaker"]
                    elif right_dur > left_dur and right_dur >= 0:
                        cur["speaker"] = right["speaker"]
    segs = _merge_adjacent_same_speaker(segs)
    return segs

# --- GPU VAD: ダイアライゼーションから音声区間を抽出 ---
def extract_speech_intervals_from_diarization(diarization_result, min_gap: float, max_end: float) -> List[Dict[str, float]]:
    """
    pyannoteの話者ターンを統合し、音声アクティビティ区間のリストを返す。
    min_gap 以下の隙間は結合する。
    """
    turns = []
    for turn, _, _ in diarization_result.itertracks(yield_label=True):
        s = max(0.0, float(turn.start))
        e = float(turn.end)
        if max_end is not None:
            e = min(e, max_end)
        if e > s:
            turns.append((s, e))
    if not turns:
        return []
    turns.sort(key=lambda x: x[0])
    merged = []
    cur_s, cur_e = turns[0]
    for s, e in turns[1:]:
        if s - cur_e <= min_gap:  # 近接・短隙間は連結
            cur_e = max(cur_e, e)
        else:
            merged.append({"start": cur_s, "end": cur_e})
            cur_s, cur_e = s, e
    merged.append({"start": cur_s, "end": cur_e})
    return merged

class _TmpWord:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end

class _TmpSegment:
    def __init__(self, start: float, end: float, text: str, words: Optional[List[_TmpWord]]):
        self.start = start
        self.end = end
        self.text = text
        self.words = words or []

# --- RunPodハンドラ ---
def handler(job):
    """
    RunPod Serverlessのエントリーポイント。
    R2から音声ファイルをダウンロードし、文字起こしと話者分離を実行する。
    """
    try:
        # モデルとS3クライアントが初期化されていることを確認
        ensure_models_loaded()
        if _s3_client is None:
            init_s3_client()
            if _s3_client is None:
                return {"error": "R2クライアントの初期化に失敗しました。環境変数を確認してください。"}

        params = job.get("input", {})
        
        # R2のオブジェクトキーを受け取る
        object_key = params.get("object_key")
        if not object_key:
            return {"error": "入力に 'object_key' がありません。"}

        bucket_name = os.environ["R2_BUCKET_NAME"]

        # パラメータの取得
        language = params.get("language", DEFAULT_LANGUAGE)
        overlap_ratio_threshold = float(params.get("overlap_ratio_threshold", 0.3))
        merge_short_threshold = float(params.get("merge_short_segment_threshold", 0.5))
        speaker_hold_time = float(params.get("speaker_hold_time", 0.8))

        # R2からファイルをダウンロードするための一時ファイルを作成
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            logging.info(f"[Job] Downloading audio from R2: s3://{bucket_name}/{object_key}...")
            _s3_client.download_file(bucket_name, object_key, tmp_path)
            logging.info(f"[Job] Audio downloaded to {tmp_path}")
        except ClientError as e:
            logging.error(f"R2からのダウンロードに失敗しました: {e}")
            return {"error": f"R2からのダウンロードに失敗しました: {e.response['Error']['Message']}"}

        try:
            waveform, sample_rate = torchaudio.load(tmp_path)
            orig_channels = waveform.shape[0] if waveform.dim() == 2 else 1
            # 前処理: モノラル・16kHzへ変換
            proc_waveform, proc_sample_rate = preprocess_audio_to_mono_16k(waveform, sample_rate, target_rate=16000)
            audio_duration = proc_waveform.shape[0] / proc_sample_rate
            logging.info(f"[Job] Audio loaded. orig_sr={sample_rate}, orig_channels={orig_channels}, proc_sr={proc_sample_rate}, duration_sec={audio_duration:.2f}")
        except Exception as e:
            return {"error": f"音声ファイルの読み込みに失敗しました: {e}"}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        logging.info("[Job] Diarization...")
        diarization_input = {"waveform": proc_waveform.unsqueeze(0), "sample_rate": proc_sample_rate}
        # オプションの話者数パラメータを受け取りパイプラインに渡す
        if params.get("num_speakers") is not None:
            diarization_input["num_speakers"] = int(params["num_speakers"])
        if params.get("min_speakers") is not None:
            diarization_input["min_speakers"] = int(params["min_speakers"])
        if params.get("max_speakers") is not None:
            diarization_input["max_speakers"] = int(params["max_speakers"])
        diarization_result = _diarization_pipeline(diarization_input)

        logging.info("[Job] Transcription...")
        audio_np = proc_waveform.numpy()
        vad_mode = str(params.get("vad_mode", "builtin")).lower()  # builtin | gpu
        min_silence_ms = int(params.get("min_silence_duration_ms", 250))
        if vad_mode == "gpu":
            # ダイアライゼーション結果から音声区間を抽出し、その区間ごとにWhisperを実行
            min_gap_sec = max(0.0, float(min_silence_ms) / 1000.0)
            intervals = extract_speech_intervals_from_diarization(diarization_result, min_gap=min_gap_sec, max_end=audio_duration)
            logging.info(f"[Job] GPU VAD intervals: {len(intervals)} (min_gap={min_gap_sec:.3f}s)")
            if len(intervals) == 0:
                logging.warning("[Job] No intervals from GPU VAD. Falling back to single-chunk transcription without VAD...")
                seg_iter, info = _whisper_model.transcribe(
                    audio_np,
                    beam_size=5,
                    language=language if language and language.lower()!= "none" else None,
                    vad_filter=False,
                    word_timestamps=True,
                )
                segments = list(seg_iter)
                logging.info(f"[Job] Fallback segments (VAD=False): {len(segments)}")
            else:
                collected_segments = []
                info = None
                for iv in intervals:
                    s_sec = max(0.0, float(iv["start"]))
                    e_sec = min(audio_duration, float(iv["end"]))
                    if e_sec <= s_sec:
                        continue
                    s_idx = int(round(s_sec * proc_sample_rate))
                    e_idx = int(round(e_sec * proc_sample_rate))
                    chunk = audio_np[s_idx:e_idx]
                    if chunk.size == 0:
                        continue
                    seg_iter, info_local = _whisper_model.transcribe(
                        chunk,
                        beam_size=5,
                        language=language if language and language.lower()!= "none" else None,
                        vad_filter=False,
                        word_timestamps=True,
                    )
                    info = info or info_local
                    for seg in seg_iter:
                        # 時刻を元音声のオフセットに合わせて補正
                        offset = s_sec
                        adj_start = float(seg.start) + offset
                        adj_end = float(seg.end) + offset
                        # 単語も補正
                        words = []
                        if getattr(seg, "words", None):
                            for w in seg.words:
                                ws = getattr(w, "start", None)
                                we = getattr(w, "end", None)
                                if ws is None or we is None:
                                    continue
                                words.append(_TmpWord(float(ws) + offset, float(we) + offset))
                        collected_segments.append(_TmpSegment(adj_start, adj_end, seg.text.strip(), words))
                # Whisperの内蔵VADを使わなかったので、そのまま使用
                segments = collected_segments
                logging.info(f"[Job] Transcription segments (GPU VAD, merged intervals={len(intervals)}): {len(segments)}")
        else:
            # 従来のWhisper内蔵VADを利用
            use_vad = bool(params.get("use_vad", True))
            segments_iter, info = _whisper_model.transcribe(
                audio_np,
                beam_size=5,
                language=language if language and language.lower()!= "none" else None,
                vad_filter=use_vad,
                vad_parameters=dict(min_silence_duration_ms=min_silence_ms),
                word_timestamps=True,
            )
            segments = list(segments_iter)
            logging.info(f"[Job] Transcription segments (with VAD={use_vad}, silence_ms={min_silence_ms}): {len(segments)}")

            # VAD有効でセグメントが0の場合、VAD無しで再試行
            if len(segments) == 0 and use_vad:
                logging.warning("[Job] No segments detected with VAD. Retrying without VAD...")
                segments_iter, info = _whisper_model.transcribe(
                    audio_np,
                    beam_size=5,
                    language=language if language and language.lower()!= "none" else None,
                    vad_filter=False,
                    word_timestamps=True,
                )
                segments = list(segments_iter)
                logging.info(f"[Job] Transcription segments after retry (VAD=False): {len(segments)}")

        logging.info("[Job] Combine + speaker assignment...")
        combined = []
        prev = "UNKNOWN_SPEAKER"
        for seg in segments:
            s_val = float(getattr(seg, "start", 0.0))
            e_val = float(getattr(seg, "end", s_val))
            if s_val > audio_duration:
                logging.warning(f"Skipping segment beyond audio duration: start={s_val:.2f}s > {audio_duration:.2f}s")
                continue
            spk = determine_speaker(seg, diarization_result, prev, overlap_ratio_threshold)
            combined.append({
                "speaker": spk,
                "start": s_val,
                "end": e_val,
                "text": getattr(seg, "text", "").strip()
            })
            prev = spk

        smoothed = smooth_labels(combined, merge_short_threshold, speaker_hold_time)
        final = [{
            "speaker": s["speaker"],
            "start_time": format_timestamp(s["start"]),
            "end_time": format_timestamp(s["end"]),
            "text": s.get("text", "")
        } for s in smoothed]

        return {
            "language": getattr(info, "language", language),
            "segments": final
        }
    except Exception as e:
        import traceback
        logging.error(f"Error during job execution: {e}")
        traceback.print_exc()
        return {"error": f"処理中に予期せぬ例外が発生しました: {e}"}


# --- サーバー起動 ---
if __name__ == "__main__":
    # ワーカー起動時にモデルとクライアントを初期化
    init_s3_client()
    ensure_models_loaded()
    runpod.serverless.start({"handler": handler})