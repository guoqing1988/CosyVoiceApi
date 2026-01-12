from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from config import settings
from models import load_cosyvoice_model
from schemas import TTSRequest
from utils import wav_to_base64

app = FastAPI(title="CosyVoiceAPI")

@app.on_event("startup")
async def startup_event():
    app.state.cosy_model = load_cosyvoice_model(settings.MODEL_DIR)

@app.get("/")
def root():
    return {"status": "ok", "message": "CosyVoiceAPI Running"}

# -------- 文本到语音 API --------

@app.post("/tts")
async def tts(req: TTSRequest):
    model = app.state.cosy_model
    wav = await run_in_threadpool(model.infer, req.text)
    b64 = wav_to_base64(wav, model.sample_rate)
    return JSONResponse({"audio": b64})

# -------- HTTP 流式输出 TTS --------

@app.post("/tts/stream")
async def tts_stream(req: TTSRequest):
    model = app.state.cosy_model

    def generator():
        for chunk in model.infer_stream(req.text):
            yield chunk.tobytes()

    return StreamingResponse(generator(), media_type="audio/wav")

# -------- WebSocket 异步流式 TTS --------

@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    model = app.state.cosy_model
    try:
        while True:
            data = await ws.receive_json()
            text = data.get("text", "")
            # 字符流式反馈
            for chunk in model.infer_stream(text):
                await ws.send_bytes(chunk.tobytes())
            await ws.send_json({"done": True})
    except Exception:
        await ws.close()

# -------- 文本到 llm 文本生成（vLLM 示例） --------

@app.post("/llm")
async def llm(req: Request):
    body = await req.json()
    prompt = body.get("prompt", "")
    from vllm_engine import init_vllm, stream_tokens

    engine = init_vllm(settings.MODEL_DIR + "/vllm")  # 指向 vllm 子模型
    async def event_stream():
        async for token in stream_tokens(prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
