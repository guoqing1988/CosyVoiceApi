from fastapi import APIRouter, WebSocket, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import logging

from ..models import get_cosy_model
from ..config import settings
from ..schemas import TTSRequest
from ..services import TTSService
from ..utils import wav_to_base64, get_exception_error

router = APIRouter()
logger = logging.getLogger(__name__)

async def _async_audio_generator(req: TTSRequest):
    """异步音频生成器 (用于 WebSocket)"""
    for chunk in TTSService.generate_audio_stream(req):
        yield chunk

@router.post("/v1/tts", tags=["TTS"])
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


@router.post("/v1/tts/stream", tags=["TTS"])
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

@router.websocket("/ws/v1/tts")
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
