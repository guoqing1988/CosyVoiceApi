"""
TTS 生成服务
封装 TTS 业务逻辑,包括音频生成、采样率处理、性能监控等
"""
from typing import Generator, Optional, Dict
import logging
import torch
import random

from cosyvoice.utils.common import set_all_random_seed

from ..models import get_cosy_model, get_inference_lock
from ..schemas import TTSRequest
from ..utils import resample_audio, PerformanceMonitor
from ..config import settings
from .voice_service import VoiceService

logger = logging.getLogger(__name__)


class TTSService:
    """TTS 生成服务"""
    
    @staticmethod
    def generate_audio_stream(
        req: TTSRequest,
        enable_monitor: bool = True
    ) -> Generator[bytes, None, None]:
        """
        生成音频流
        
        Args:
            req: TTS 请求参数
            enable_monitor: 是否启用性能监控
        
        Yields:
            音频数据块 (bytes)
        """
        model = get_cosy_model()
        if not model:
            raise RuntimeError("模型未加载")
        
        # 设置随机种子
        if req.seed is not None:
            set_all_random_seed(req.seed)
        else:
            seed = random.randint(1, 100000000)
            set_all_random_seed(seed)
        
        # 性能监控
        monitor = None
        if enable_monitor and settings.ENABLE_PERFORMANCE_MONITOR:
            monitor = PerformanceMonitor(f"TTS-{req.mode}")
            monitor.start()
        
        # 处理 voice_id (优先级高于 speaker)
        prompt_wav_path = req.prompt_wav_path
        prompt_text = req.prompt_text
        zero_shot_spk_id = ""
        voice_id = req.voice_id or req.speaker
        if voice_id and req.mode in ["sft", "zero_shot"]:
            voice_info = VoiceService.get_voice_by_id(voice_id)
            if voice_info:
                prompt_wav_path = voice_info["file"]
                prompt_text = voice_info["prompt_text"]
                zero_shot_spk_id = voice_id
                req.mode = "zero_shot"
                logger.info(f"使用预加载音色: {voice_id}")
            else:
                logger.warning(f"音色 ID '{voice_id}' 不存在,使用默认配置")
        
        # 根据模式生成音频
        try:
            # 使用并发锁保护推理过程
            lock = get_inference_lock()
            with lock:
                audio_iterator = TTSService._get_audio_iterator(
                    model, req, prompt_wav_path, prompt_text, zero_shot_spk_id
                )
                
                first_chunk = True
                for chunk in audio_iterator:
                    audio_tensor = chunk['tts_speech']
                    
                    # 采样率重采样
                    if settings.OUTPUT_SAMPLE_RATE != model.sample_rate:
                        audio_tensor = resample_audio(
                            audio_tensor,
                            model.sample_rate,
                            settings.OUTPUT_SAMPLE_RATE
                        )
                    
                    # 转换为字节流
                    audio_data = audio_tensor.numpy().flatten().tobytes()
                    
                    # 性能监控
                    if monitor:
                        if first_chunk:
                            monitor.record_first_chunk()
                            first_chunk = False
                        monitor.record_chunk(len(audio_data))
                    
                    yield audio_data
                
        except Exception as e:
            logger.error(f"音频生成失败: {e}")
            raise
        
        finally:
            if monitor:
                monitor.finish()
    
    @staticmethod
    def _get_audio_iterator(
        model,
        req: TTSRequest,
        prompt_wav_path: str,
        prompt_text: str,
        zero_shot_spk_id: Optional[str]
    ):
        """
        根据模式获取音频迭代器
        
        Args:
            model: CosyVoice 模型
            req: TTS 请求
            prompt_wav_path: 参考音频路径
            prompt_text: 参考文本
            zero_shot_spk_id: Zero-shot 音色 ID
        
        Returns:
            音频迭代器
        """
        if req.mode == "sft":
            return model.inference_sft(
                req.text,
                req.speaker,
                stream=req.stream,
                speed=req.speed
            )
        
        elif req.mode == "zero_shot":
            # 格式化 prompt_text
            if not prompt_text.startswith("You are a helpful assistant.<|endofprompt|>"):
                formatted_prompt = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
            else:
                formatted_prompt = prompt_text
            
            return model.inference_zero_shot(
                req.text,
                formatted_prompt,
                prompt_wav_path,
                stream=req.stream,
                speed=req.speed,
                zero_shot_spk_id=zero_shot_spk_id
            )
        
        elif req.mode == "cross_lingual":
            return model.inference_cross_lingual(
                req.text,
                prompt_wav_path,
                stream=req.stream,
                speed=req.speed
            )
        
        elif req.mode == "instruct":
            if hasattr(model, 'inference_instruct2'):
                return model.inference_instruct2(
                    req.text,
                    req.instruct_text,
                    prompt_wav_path,
                    stream=req.stream,
                    speed=req.speed
                )
            else:
                return model.inference_instruct(
                    req.text,
                    req.speaker,
                    req.instruct_text,
                    stream=req.stream,
                    speed=req.speed
                )
        
        elif req.mode == "vc":
            return model.inference_vc(
                req.source_wav_path,
                prompt_wav_path,
                stream=req.stream,
                speed=req.speed
            )
        
        else:
            raise ValueError(f"不支持的模式: {req.mode}")
    
    @staticmethod
    def generate_audio_complete(req: TTSRequest) -> tuple[torch.Tensor, int, Dict]:
        """
        生成完整音频 (非流式)
        
        Args:
            req: TTS 请求参数
        
        Returns:
            (audio_tensor, sample_rate, performance_stats)
        """
        model = get_cosy_model()
        if not model:
            raise RuntimeError("模型未加载")
        
        # 强制设置为非流式
        req.stream = False
        
        # 收集所有音频块
        audio_chunks = []
        monitor = None
        
        if settings.ENABLE_PERFORMANCE_MONITOR:
            monitor = PerformanceMonitor(f"TTS-{req.mode}-Complete")
            monitor.start()
        
        for chunk_bytes in TTSService.generate_audio_stream(req, enable_monitor=True):
            # 将 bytes 转回 tensor (假设是 float32)
            import numpy as np
            chunk_array = np.frombuffer(chunk_bytes, dtype=np.float32)
            chunk_tensor = torch.from_numpy(chunk_array)
            audio_chunks.append(chunk_tensor)
            
            if monitor:
                monitor.record_chunk(len(chunk_bytes))
        
        # 合并所有音频块
        if not audio_chunks:
            raise RuntimeError("音频生成失败,无数据返回")
        
        full_audio = torch.cat(audio_chunks, dim=0)
        
        stats = {}
        if monitor:
            stats = monitor.finish()
        
        return full_audio, settings.OUTPUT_SAMPLE_RATE, stats
