from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import torch
import os
import logging

from .config import settings
from .models import load_cosyvoice_model, get_cosy_model
from .schemas import (
    TTSRequest, 
    HealthResponse, 
    VoiceListResponse,
    TTSResponse
)
from .services import VoiceService, TTSService
from .utils import wav_to_base64, get_exception_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CosyVoice API",
    description="高并发语音生成 API 服务",
    version="2.0.0"
)

@app.on_event("startup")
async def startup_event():
    """应用启动事件 - 加载模型和音色"""
    logger.info("=" * 60)
    logger.info("🚀 CosyVoice API 服务启动中...")
    logger.info("=" * 60)
    
    try:
        # 加载模型 (内部会自动加载音色和预热)
        load_cosyvoice_model(
            model_dir=settings.MODEL_DIR,
            fp16=settings.FP16,
            use_vllm=settings.USE_VLLM
        )
        
        logger.info("=" * 60)
        logger.info("✅ CosyVoice API 服务启动成功!")
        logger.info(f"📍 访问地址: http://{settings.HOST}:{settings.PORT}")
        logger.info(f"📖 API 文档: http://{settings.HOST}:{settings.PORT}/docs")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ 服务启动失败: {e}")
        logger.error(get_exception_error())
        raise


# ========== 健康检查接口 ==========

@app.get("/v1/health", response_model=HealthResponse)
def health():
    """
    健康检查接口
    
    返回服务运行状态、GPU 状态、模型信息、音色数量等
    """
    model = get_cosy_model()
    
    return HealthResponse(
        status="ok" if model else "error",
        gpu=torch.cuda.is_available(),
        model=model.__class__.__name__ if model else "Not Loaded",
        sample_rate=model.sample_rate if model else None,
        output_sample_rate=settings.OUTPUT_SAMPLE_RATE,
        voice_count=VoiceService.get_voice_count(),
        vllm_enabled=settings.USE_VLLM
    )


# ========== 音色管理接口 ==========

@app.get("/v1/speakers")
def get_speakers():
    """
    获取预训练音色列表 (SFT 模式)
    
    返回模型内置的预训练音色名称
    """
    model = get_cosy_model()
    if model:
        return {"speakers": model.list_available_spks()}
    return {"speakers": []}


@app.get("/v1/voices", response_model=VoiceListResponse)
def get_voices():
    """
    获取预加载音色列表
    
    返回所有已缓存的 zero-shot 音色信息
    """
    voices = VoiceService.list_all_voices()
    
    return VoiceListResponse(
        voices=voices,
        total=len(voices),
        default_voice_id=settings.DEFAULT_VOICE_ID
    )


# ========== TTS 生成接口 ==========

@app.post("/v1/tts")
async def tts(req: TTSRequest):
    """
    统一 TTS 合成接口
    
    支持多种模式:
    - sft: 预训练音色
    - zero_shot: 3s 极速复刻
    - cross_lingual: 跨语种复刻
    - instruct: 自然语言控制
    - vc: 声音转换
    
    支持流式和非流式返回
    """
    model = get_cosy_model()
    if not model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    try:
        if req.stream:
            # 流式返回
            return StreamingResponse(
                TTSService.generate_audio_stream(req),
                media_type="audio/pcm",
                headers={
                    "X-Sample-Rate": str(settings.OUTPUT_SAMPLE_RATE),
                    "X-Channels": "1",
                    "X-Bits": "16"
                }
            )
        else:
            # 非流式返回
            full_audio, sample_rate, stats = TTSService.generate_audio_complete(req)
            
            # 转换为 Base64
            b64 = wav_to_base64(full_audio.numpy(), sample_rate)
            
            response_data = {
                "audio": b64,
                "sample_rate": sample_rate
            }
            
            # 如果启用性能监控,添加性能指标
            if settings.ENABLE_PERFORMANCE_MONITOR and stats:
                response_data["performance"] = stats
            
            return JSONResponse(response_data)
            
    except Exception as e:
        logger.error(f"TTS 生成失败: {e}")
        logger.error(get_exception_error())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/tts/stream")
async def tts_stream(req: TTSRequest):
    """
    流式 TTS 合成接口
    
    强制启用流式返回,其他参数同 /v1/tts
    """
    req.stream = True
    model = get_cosy_model()
    if not model:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    return StreamingResponse(
        TTSService.generate_audio_stream(req),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(settings.OUTPUT_SAMPLE_RATE),
            "X-Channels": "1",
            "X-Bits": "16"
        }
    )


# ========== WebSocket 实时接口 ==========

@app.websocket("/ws/v1/tts")
async def websocket_tts(ws: WebSocket):
    """
    WebSocket 实时语音合成
    
    客户端发送 JSON 配置,服务端推送音频二进制数据
    """
    await ws.accept()
    model = get_cosy_model()
    
    if not model:
        await ws.send_json({"error": "模型未加载"})
        await ws.close()
        return
    
    try:
        while True:
            # 接收请求
            data = await ws.receive_json()
            req = TTSRequest(**data)
            
            # 生成并推送音频
            async for chunk_bytes in _async_audio_generator(req):
                await ws.send_bytes(chunk_bytes)
            
            # 发送完成信号
            await ws.send_json({"done": True})
            
    except Exception as e:
        logger.error(f"WebSocket 错误: {e}")
        await ws.send_json({"error": str(e)})
        await ws.close()


async def _async_audio_generator(req: TTSRequest):
    """异步音频生成器 (用于 WebSocket)"""
    for chunk in TTSService.generate_audio_stream(req):
        yield chunk


# ========== 静态文件服务 ==========

# 挂载静态文件 - 必须放在最后以避免遮蔽 API 路由
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    logger.info(f"📁 静态文件目录: {static_dir}")
