import os
import torch
import runpod
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import tempfile
import base64
import torchaudio

# --- グローバル設定とモデルのロード ---
# このセクションはワーカー起動時に一度だけ実行されます

# Dockerfileで焼き付けたWhisperモデルへのコンテナ内パスを定義
FASTER_WHISPER_PATH = "/app/models/Systran/faster-whisper-large-v2"

# Hugging Faceの認証トークンを環境変数から取得
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("警告: 環境変数 'HF_TOKEN' が設定されていません。gatedモデルの読み込みに失敗する可能性があります。")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if torch.cuda.is_available() else "int8"

print(f"INFO: キャッシュされたモデルを読み込みます...")
print(f"INFO: デバイス: {DEVICE}, 計算タイプ: {COMPUTE_TYPE} を使用します。")

# 1. ローカルパスから文字起こしモデルをロード
# faster-whisperはモデルディレクトリを直接指定する方式
model = WhisperModel(
    FASTER_WHISPER_PATH,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

# 2. リポジトリIDを指定して話者分離モデルをロード
# from_pretrainedは、まずローカルキャッシュ（/app/models）を探す
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
diarization_pipeline.to(torch.device(DEVICE))

print("INFO: 全てのモデルが正常に読み込まれました。")

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