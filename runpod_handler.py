import os
import torch
import torchaudio
import runpod
from typing import List, Dict, Any
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import traceback

# --- グローバル設定とモデルのロード ---
# この部分はコンテナ起動時に一度だけ実行される
HF_AUTH_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

print(f"INFO: Handler loading models to {DEVICE} with compute type {COMPUTE_TYPE}...")

# 1. 文字起こしモデル (faster-whisper)
model = WhisperModel("large-v3", device=DEVICE, compute_type=COMPUTE_TYPE)

# 2. 話者分離モデル (pyannote.audio)
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_AUTH_TOKEN,
)
diarization_pipeline.to(torch.device(DEVICE))

print("INFO: Models loaded successfully.")

# --- ヘルパー関数 (変更なし) ---
def format_timestamp(seconds: float) -> str:
    assert seconds >= 0, "Negative timestamp not supported"
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_val, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d},{milliseconds:03d}"

def assign_speaker_to_segment(segment: dict, diarization_result: "pyannote.core.Annotation") -> str:
    max_overlap, best_speaker = 0, "UNKNOWN"
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        overlap = max(0, min(segment["end"], turn.end) - max(segment["start"], turn.start))
        if overlap > max_overlap:
            max_overlap, best_speaker = overlap, speaker
    return best_speaker

# --- RunPodハンドラ関数 ---
def handler(job):
    """
    RunPodがリクエストを受け取ったときに呼び出すメイン関数
    """
    job_input = job['input']
    
    # Base64エンコードされた音声データとファイル名を取得
    audio_base64 = job_input.get("audio_base64")
    file_name = job_input.get("file_name", "audio.tmp")
    language = job_input.get("language", "ja")
    
    if not audio_base64:
        return {"error": "audio_base64 not provided"}
        
    import base64
    audio_data = base64.b64decode(audio_base64)
    
    # 一時ファイルに音声データを書き込む
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file_name}") as tmp_file:
        tmp_file.write(audio_data)
        tmp_audio_path = tmp_file.name

    try:
        # 文字起こし処理
        waveform, sample_rate = torchaudio.load(tmp_audio_path)
        diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})
        lang_option = language if language != "auto" else None
        segments, _ = model.transcribe(tmp_audio_path, language=lang_option, beam_size=5, vad_filter=True)
        
        # 結果の整形
        final_results = []
        for segment in segments:
            speaker = assign_speaker_to_segment({"start": segment.start, "end": segment.end}, diarization_result)
            final_results.append({
                "speaker": speaker,
                "start_time": format_timestamp(segment.start),
                "end_time": format_timestamp(segment.end),
                "text": segment.text.strip(),
            })
        
        return final_results

    except Exception as e:
        # エラーが発生した場合、詳細なスタックトレースをログに出力する
        error_trace = traceback.format_exc()
        print(f"ERROR: An exception occurred: {e}")
        print(f"TRACEBACK: {error_trace}")
        return {"error": str(e), "traceback": error_trace} # <- レスポンスにもトレースバックを含める
    finally:
        os.unlink(tmp_audio_path)

# RunPodワーカーを開始
runpod.serverless.start({"handler": handler})