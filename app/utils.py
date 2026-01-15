import base64
import soundfile as sf
import io
import time
import logging
import torch
import torchaudio
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

def wav_to_base64(wav: Union[np.ndarray, torch.Tensor], sample_rate: int = 22050) -> str:
    """
    将音频数据转换为 Base64 编码的 WAV 格式
    
    Args:
        wav: 音频数据 (numpy array 或 torch tensor)
        sample_rate: 采样率
    
    Returns:
        Base64 编码的字符串
    """
    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

def get_exception_error() -> str:
    """获取异常堆栈信息"""
    import traceback
    return traceback.format_exc()

def resample_audio(
    audio: torch.Tensor, 
    orig_sr: int, 
    target_sr: int
) -> torch.Tensor:
    """
    音频重采样
    
    Args:
        audio: 输入音频 tensor, shape: (channels, samples) 或 (samples,)
        orig_sr: 原始采样率
        target_sr: 目标采样率
    
    Returns:
        重采样后的音频 tensor
    """
    if orig_sr == target_sr:
        return audio
    
    # 确保是 2D tensor
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    resampler = torchaudio.transforms.Resample(
        orig_freq=orig_sr,
        new_freq=target_sr
    )
    
    resampled = resampler(audio)
    return resampled

def load_audio_file(file_path: str) -> tuple[torch.Tensor, int]:
    """
    加载音频文件
    
    Args:
        file_path: 音频文件路径
    
    Returns:
        (audio_tensor, sample_rate)
    """
    try:
        audio, sr = torchaudio.load(file_path)
        return audio, sr
    except Exception as e:
        logger.error(f"加载音频文件失败: {file_path}, 错误: {e}")
        raise

class PerformanceMonitor:
    """性能监控工具"""
    
    def __init__(self, task_name: str = "Task"):
        self.task_name = task_name
        self.start_time = None
        self.first_chunk_time = None
        self.total_bytes = 0
        self.chunk_count = 0
    
    def start(self):
        """开始计时"""
        self.start_time = time.time()
        self.first_chunk_time = None
        self.total_bytes = 0
        self.chunk_count = 0
    
    def record_first_chunk(self):
        """记录首帧时间"""
        if self.first_chunk_time is None and self.start_time is not None:
            self.first_chunk_time = time.time()
            latency_ms = (self.first_chunk_time - self.start_time) * 1000
            logger.info(f"⚡ {self.task_name} 首帧延迟: {latency_ms:.0f}ms")
            return latency_ms
        return None
    
    def record_chunk(self, chunk_size: int):
        """记录数据块"""
        self.total_bytes += chunk_size
        self.chunk_count += 1
    
    def finish(self) -> dict:
        """完成并返回统计信息"""
        if self.start_time is None:
            return {}
        
        total_time_ms = (time.time() - self.start_time) * 1000
        first_chunk_latency_ms = 0
        
        if self.first_chunk_time:
            first_chunk_latency_ms = (self.first_chunk_time - self.start_time) * 1000
        
        stats = {
            "total_time_ms": round(total_time_ms, 2),
            "first_chunk_latency_ms": round(first_chunk_latency_ms, 2),
            "total_bytes": self.total_bytes,
            "total_kb": round(self.total_bytes / 1024, 2),
            "chunk_count": self.chunk_count
        }
        
        logger.info(
            f"✅ {self.task_name} 完成: "
            f"总耗时 {stats['total_time_ms']:.0f}ms, "
            f"数据量 {stats['total_kb']:.1f}KB, "
            f"分块数 {stats['chunk_count']}"
        )
        
        return stats