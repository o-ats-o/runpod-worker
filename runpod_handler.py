import runpod
import os
import time

# モデルロードなど、重い処理はすべてコメントアウト
print("INFO: Handler script started.")
print(f"HUGGING_FACE_TOKEN is set: {'HUGGING_FACE_TOKEN' in os.environ}") # トークンが設定されているか確認

def handler(job):
    """
    リクエストを受け取ったことを確認するためのダミーハンドラ
    """
    print("INFO: Handler function was called!")
    
    # 受け取った入力をログに出力
    job_input = job.get('input', {})
    print(f"INFO: Received input: {job_input}")
    
    # 成功したことを示す簡単な応答を返す
    return {
        "status": "success",
        "message": "Dummy handler executed correctly!",
        "timestamp": time.time()
    }

print("INFO: Starting RunPod serverless worker...")
runpod.serverless.start({"handler": handler})