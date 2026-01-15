from fastapi import APIRouter
from ..models import get_cosy_model
from ..schemas import VoiceListResponse
from ..config import settings
from ..services import VoiceService

router = APIRouter()

@router.get("/v1/speakers", tags=["Voice"])
def get_speakers():
    """
    获取预训练音色列表 (SFT 模式)
    
    返回模型内置的预训练音色名称
    """
    model = get_cosy_model()
    if model:
        return {"speakers": model.list_available_spks()}
    return {"speakers": []}


@router.get("/v1/voices", response_model=VoiceListResponse, tags=["Voice"])
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
