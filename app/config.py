import os
from typing import List, Dict
from dataclasses import dataclass
from pydantic_settings import BaseSettings

@dataclass
class VoiceConfig:
    """音色配置数据类"""
    id: str  # 音色唯一标识
    file: str  # 音频文件名 (相对于 ASSET_DIR)
    prompt_text: str  # 参考音频对应的文本
    description: str = ""  # 音色描述

class Settings(BaseSettings):
    # ========== 模型配置 ==========
    MODEL_DIR: str = os.getenv("COSYVOICE_MODEL_DIR", "/data/models/cosyvoice")
    USE_VLLM: bool = True  # 是否启用 vLLM 加速
    FP16: bool = True  # 是否使用 FP16 推理
    
    # ========== 服务配置 ==========
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_WORKERS: int = 4
    
    # ========== 音频配置 ==========
    OUTPUT_SAMPLE_RATE: int = 24000  # 输出采样率: 16000 或 24000
    
    # ========== 音色配置 ==========
    ASSET_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "asset"
    )  # 音色文件目录
    
    ENABLE_MODEL_WARMUP: bool = True  # 是否启用模型预热
    DEFAULT_VOICE_ID: str = "default"  # 默认音色 ID
    
    # 预定义音色配置列表
    VOICE_CONFIGS: List[Dict] = [
        {
            "id": "default",
            "file": "zero_shot_prompt.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            "description": "默认女声",
        },
        {
            "id": "tone_man",
            "file": "tone_man.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>真不好意思，从小至今，他还从来没有被哪一位异性朋友亲吻过呢。",
            "description": "男声",
        },
        {
            "id": "tone_woman",
            "file": "tone_woman.wav", 
            "prompt_text": "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。",
            "description": "女声1",
        },
        {
            "id": "tone_woman2",
            "file": "tone_woman2.wav",
            "prompt_text": "You are a helpful assistant.<|endofprompt|>您好，我是智能电话助手，很高兴为您服务。请问您需要咨询业务预约办理还是查询信息？",
            "description": "女声2",
        }
    ]
    
    # ========== 性能配置 ==========
    ENABLE_PERFORMANCE_MONITOR: bool = True  # 启用性能监控
    LOG_FIRST_CHUNK_LATENCY: bool = True  # 记录首帧延迟

settings = Settings()
