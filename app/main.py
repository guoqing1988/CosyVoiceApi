from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import io
import torch
import torchaudio
import numpy as np
import os

from .config import settings
from .models import load_cosyvoice_model, get_cosy_model
from .schemas import TTSRequest
from .utils import wav_to_base64,get_exception_error

app = FastAPI(title="CosyVoiceAPI")

@app.on_event("startup")
async def startup_event():
    load_cosyvoice_model(settings.MODEL_DIR)

@app.get("/v1/health")
def health():
    model = get_cosy_model()
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "model": model.__class__.__name__ if model else "Not Loaded"
    }

@app.get("/v1/speakers")
def get_speakers():
    model = get_cosy_model()
    if model:
        return {"speakers": model.list_available_spks()}
    return {"speakers": []}

async def generate_audio_chunks(model, req: TTSRequest):
    """Unified generator for all inference modes"""
    try:
        if req.mode == "sft":
            it = model.inference_sft(req.text, req.speaker, stream=req.stream, speed=req.speed)
        elif req.mode == "zero_shot":
            formatted_tts_text = f"You are a helpful assistant.<|endofprompt|>{req.prompt_text}"
            it = model.inference_zero_shot(req.text, formatted_tts_text, req.prompt_wav_path, stream=req.stream, speed=req.speed)
        elif req.mode == "cross_lingual":
            it = model.inference_cross_lingual(req.text, req.prompt_wav_path, stream=req.stream, speed=req.speed)
        elif req.mode == "instruct":
            if hasattr(model, 'inference_instruct2'):
                it = model.inference_instruct2(req.text, req.instruct_text, req.prompt_wav_path, stream=req.stream, speed=req.speed)
            else:
                it = model.inference_instruct(req.text, req.speaker, req.instruct_text, stream=req.stream, speed=req.speed)
        elif req.mode == "vc":
            it = model.inference_vc(req.source_wav_path, req.prompt_wav_path, stream=req.stream, speed=req.speed)
        else:
            raise ValueError(f"Unknown mode: {req.mode}")

        for chunk in it:
            audio_data = chunk['tts_speech'].numpy().tobytes()
            yield audio_data
    except Exception as e:
        print(f"Error in generation: {e}")
        yield b""

@app.post("/v1/tts")
async def tts(req: TTSRequest):
    model = get_cosy_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.stream:
        audio_data_list = []
        try:
            if req.mode == "sft":
                it = model.inference_sft(req.text, req.speaker, stream=False, speed=req.speed)
            elif req.mode == "zero_shot":
                formatted_tts_text = f"You are a helpful assistant.<|endofprompt|>{req.prompt_text}"
                it = model.inference_zero_shot(req.text, formatted_tts_text, req.prompt_wav_path, stream=False, speed=req.speed)
            elif req.mode == "cross_lingual":
                it = model.inference_cross_lingual(req.text, req.prompt_wav_path, stream=False, speed=req.speed)
            elif req.mode == "instruct":
                if hasattr(model, 'inference_instruct2'):
                    it = model.inference_instruct2(req.text, req.instruct_text, req.prompt_wav_path, stream=False, speed=req.speed)
                else:
                    it = model.inference_instruct(req.text, req.speaker, req.instruct_text, stream=False, speed=req.speed)
            elif req.mode == "vc":
                it = model.inference_vc(req.source_wav_path, req.prompt_wav_path, stream=False, speed=req.speed)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown mode: {req.mode}")

            for chunk in it:
                audio_data_list.append(chunk['tts_speech'])

            if not audio_data_list:
                raise HTTPException(status_code=500, detail="Failed to generate audio")

            full_audio = torch.cat(audio_data_list, dim=1)
            # Flatten audio to 1D array for soundfile compatibility
            b64 = wav_to_base64(full_audio.numpy().flatten(), model.sample_rate)
            return JSONResponse({"audio": b64, "sample_rate": model.sample_rate})
        except Exception as e:
            print(get_exception_error())
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return StreamingResponse(generate_audio_chunks(model, req), media_type="audio/pcm")

@app.post("/v1/tts/stream")
async def tts_stream(req: TTSRequest):
    req.stream = True
    model = get_cosy_model()
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return StreamingResponse(generate_audio_chunks(model, req), media_type="audio/pcm")

@app.websocket("/ws/v1/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    model = get_cosy_model()
    try:
        while True:
            data = await ws.receive_json()
            req = TTSRequest(**data)
            async for chunk_bytes in generate_audio_chunks(model, req):
                await ws.send_bytes(chunk_bytes)
            await ws.send_json({"done": True})
    except Exception as e:
        print(f"WS Error: {e}")
        await ws.close()

# Mount static files - Must be at the end to not shadow API routes
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
