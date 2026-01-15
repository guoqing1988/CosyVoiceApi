from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict

class SFTRequest(BaseModel):
    text: str
    speaker: str
    stream: bool = False
    speed: float = 1.0

class ZeroShotRequest(BaseModel):
    text: str
    prompt_text: str
    prompt_wav_path: Optional[str] = None
    stream: bool = False
    speed: float = 1.0

class CrossLingualRequest(BaseModel):
    text: str
    prompt_wav_path: str
    stream: bool = False
    speed: float = 1.0

class InstructRequest(BaseModel):
    text: str
    speaker: str
    instruct_text: str
    stream: bool = False
    speed: float = 1.0

class VCRequest(BaseModel):
    source_wav_path: str
    prompt_wav_path: str
    stream: bool = False
    speed: float = 1.0

class TTSRequest(BaseModel):
    """统一 TTS 请求模型"""
    text: str = Field(..., description="要合成的文本")
    mode: str = Field(default="sft", description="合成模式: sft, zero_shot, cross_lingual, instruct, vc")
    
    # 音色相关
    speaker: Optional[str] = Field(default="中文女", description="预训练音色名称 (sft/instruct 模式)")
    voice_id: Optional[str] = Field(default=None, description="预加载音色 ID (优先级高于 speaker)")
    
    # Zero-shot / Cross-lingual 相关
    prompt_text: Optional[str] = Field(default="", description="参考音频对应的文本")
    prompt_wav_path: Optional[str] = Field(default="", description="参考音频文件路径")
    
    # Instruct 相关
    instruct_text: Optional[str] = Field(default="", description="情感/风格控制文本")
    
    # VC 相关
    source_wav_path: Optional[str] = Field(default="", description="源音频文件路径")
    
    # 生成参数
    stream: bool = Field(default=False, description="是否流式返回")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="语速: 0.5-2.0")
    seed: Optional[int] = Field(default=None, description="随机种子")

class VoiceInfo(BaseModel):
    """音色信息响应模型"""
    id: str = Field(..., description="音色唯一标识")
    file: str = Field(..., description="音频文件名")
    prompt_text: str = Field(..., description="参考文本")
    description: str = Field(default="", description="音色描述")
    is_loaded: bool = Field(default=False, description="是否已加载")

class VoiceListResponse(BaseModel):
    """音色列表响应"""
    voices: List[VoiceInfo]
    total: int
    default_voice_id: str

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    gpu: bool
    model: str
    sample_rate: Optional[int] = None
    output_sample_rate: Optional[int] = None
    voice_count: int = 0
    vllm_enabled: bool = False

class TTSResponse(BaseModel):
    """非流式 TTS 响应"""
    audio: str = Field(..., description="Base64 编码的音频数据")
    sample_rate: int = Field(..., description="采样率")
    performance: Optional[Dict] = Field(default=None, description="性能指标")
