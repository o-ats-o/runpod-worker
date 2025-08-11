import os
import torch
import runpod
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import base64
import torchaudio

# --- グローバル設定とモデルのロード ---
# このセクションはワーカー起動時に一度だけ実行

HF_AUTH_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

# 永続ボリューム上のキャッシュディレクトリを定義
CACHE_DIR = "/workspace/models_cache"

print(f"INFO: キャッシュからモデルを読み込みます: {CACHE_DIR}...")
print(f"INFO: デバイス: {DEVICE}, 計算タイプ: {COMPUTE_TYPE} を使用します。")

# 1. キャッシュから文字起こしモデルをロード
# 'download_root'パラメータで、モデルを探す/保存する場所をfaster-whisperに伝える
model = WhisperModel(
    "large-v2",
    device=DEVICE,
    compute_type=COMPUTE_TYPE,
    download_root=CACHE_DIR
)

# 2. キャッシュから話者分離モデルをロード
# 'cache_dir'パラメータで、モデルを探す/保存する場所をpyannote/huggingfaceに伝える
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_AUTH_TOKEN,
    cache_dir=CACHE_DIR
)
diarization_pipeline.to(torch.device(DEVICE))

print("INFO: 全てのモデルがキャッシュから正常に読み込まれました。")

# --- ヘルパー関数 ---

def format_timestamp(seconds: float) -> str:
    """秒を標準的なSRTタイムスタンプ形式に変換します。"""
    assert seconds >= 0, "負のタイムスタンプはサポートされていません"
    milliseconds = round(seconds * 1000.0)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds_val, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds_val:02d},{milliseconds:03d}"

def assign_speaker_to_segment(segment, diarization_result):
    """最大の重複に基づき、文字起こしセグメントに話者ラベルを割り当てます。"""
    from pyannote.core import Segment
    max_overlap = 0
    best_speaker = "UNKNOWN"
    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        # whisperセグメントと話者の発話区間の重複を計算
        overlap = max(0, min(segment.end, turn.end) - max(segment.start, turn.start))
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = speaker
    return best_speaker

# --- RunPodハンドラ ---

def handler(job):
    """各ジョブに対してRunPodから呼び出されるメイン関数です。"""
    job_input = job['input']
    audio_base64 = job_input.get("audio_base64")
    language = job_input.get("language", "ja") # 指定がない場合は日本語をデフォルトとします

    if not audio_base64:
        return {"error": "入力が見つからないか、'audio_base64'が提供されていません。"}

    audio_data = base64.b64decode(audio_base64)

    # デコードした音声を一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_data)
        tmp_audio_path = tmp_file.name

    try:
        # 1. 話者分離を実行
        print("INFO: 話者分離を実行中...")
        waveform, sample_rate = torchaudio.load(tmp_audio_path)
        diarization_result = diarization_pipeline({"waveform": waveform, "sample_rate": sample_rate})

        # 2. 文字起こしを実行
        print("INFO: 文字起こしを実行中...")
        segments, _ = model.transcribe(tmp_audio_path, language=language, beam_size=5, vad_filter=True)
        
        # 3. 結果を結合
        print("INFO: 結果を結合中...")
        final_results = []
        for segment in segments:
            speaker = assign_speaker_to_segment(segment, diarization_result)
            final_results.append({
                "speaker": speaker,
                "start_time": format_timestamp(segment.start),
                "end_time": format_timestamp(segment.end),
                "text": segment.text.strip(),
            })
        
        print("INFO: ジョブが正常に完了しました。")
        return final_results

    except Exception as e:
        print(f"ERROR: エラーが発生しました: {e}")
        return {"error": str(e)}
    finally:
        # 一時ファイルをクリーンアップ
        os.unlink(tmp_audio_path)

# RunPodサーバーレスワーカーを開始
runpod.serverless.start({"handler": handler})