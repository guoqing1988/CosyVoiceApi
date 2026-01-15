from fastapi import APIRouter
import torch
from ..models import get_cosy_model
from ..config import settings
from ..schemas import HealthResponse
from ..services import VoiceService

router = APIRouter()

@router.get("/v1/health", response_model=HealthResponse, tags=["System"])
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
