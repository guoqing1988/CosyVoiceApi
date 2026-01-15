from typing import Optional, Dict
import torch
import sys
import time
import os
import threading
import logging
from cosyvoice.cli.cosyvoice import AutoModel

from .config import settings, VoiceConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 全局变量 ==========
cosy_model: Optional[AutoModel] = None
voice_cache_manager: Optional['VoiceCacheManager'] = None
inference_lock = threading.Lock()  # 并发推理保护锁


class VoiceCacheManager:
    """音色缓存管理器"""
    
    def __init__(self, model: AutoModel):
        self.model = model
        self.voice_cache: Dict[str, Dict] = {}
        self.default_voice_id = settings.DEFAULT_VOICE_ID
    
    def load_voices(self) -> int:
        """
        批量加载音色配置
        
        Returns:
            成功加载的音色数量
        """
        logger.info(f"⚡ 正在加载 {len(settings.VOICE_CONFIGS)} 个音色配置...")
        
        loaded_count = 0
        for voice_config in settings.VOICE_CONFIGS:
            if self._load_single_voice(voice_config):
                loaded_count += 1
        
        logger.info(f"⚡ 音色加载完成,共 {loaded_count} 个可用音色: {list(self.voice_cache.keys())}")
        return loaded_count
    
    def _load_single_voice(self, voice_config: Dict) -> bool:
        """
        加载单个音色
        
        Args:
            voice_config: 音色配置字典
        
        Returns:
            是否加载成功
        """
        voice_id = voice_config["id"]
        voice_file = voice_config["file"]
        prompt_text = voice_config["prompt_text"]
        description = voice_config.get("description", "")
        
        # 查找音频文件
        voice_path = os.path.join(settings.ASSET_DIR, voice_file)
        
        if not os.path.exists(voice_path):
            logger.warning(f"❌ 音色 '{voice_id}' 的文件未找到: {voice_path}")
            return False
        
        try:
            # 使用 CosyVoice 的 add_zero_shot_spk 方法缓存音色特征
            if hasattr(self.model, 'add_zero_shot_spk'):
                self.model.add_zero_shot_spk(prompt_text, voice_path, voice_id)
            
            # 保存到本地缓存
            self.voice_cache[voice_id] = {
                "file": voice_path,
                "prompt_text": prompt_text,
                "description": description,
                "is_loaded": True
            }
            
            logger.info(f"✅ 音色 '{voice_id}' 加载成功: {voice_file}")
            return True
            
        except Exception as e:
            logger.warning(f"❌ 音色 '{voice_id}' 加载失败: {e}")
            return False
    
    def get_voice(self, voice_id: str) -> Optional[Dict]:
        """
        获取指定音色信息
        
        Args:
            voice_id: 音色 ID
        
        Returns:
            音色信息字典,不存在则返回 None
        """
        return self.voice_cache.get(voice_id)
    
    def list_voices(self) -> list:
        """
        列出所有可用音色
        
        Returns:
            音色信息列表
        """
        return [
            {
                "id": voice_id,
                "file": os.path.basename(info["file"]),
                "prompt_text": info["prompt_text"],
                "description": info.get("description", ""),
                "is_loaded": info.get("is_loaded", False)
            }
            for voice_id, info in self.voice_cache.items()
        ]
    
    def get_default_voice(self) -> Optional[Dict]:
        """获取默认音色"""
        return self.get_voice(self.default_voice_id)


def load_cosyvoice_model(
    model_dir: str = None,
    device: str = "cuda",
    fp16: bool = None,
    use_vllm: bool = None
) -> AutoModel:
    """
    加载 CosyVoice 模型
    
    Args:
        model_dir: 模型目录路径
        device: 运行设备
        fp16: 是否使用 FP16 推理
        use_vllm: 是否启用 vLLM 加速
    
    Returns:
        加载的模型实例
    """
    global cosy_model, voice_cache_manager
    
    if cosy_model is not None:
        logger.info("模型已加载,跳过重复加载")
        return cosy_model
    
    # 使用配置文件的默认值
    if model_dir is None:
        model_dir = settings.MODEL_DIR
    if fp16 is None:
        fp16 = settings.FP16
    if use_vllm is None:
        use_vllm = settings.USE_VLLM
    
    logger.info(f"正在加载模型: {model_dir}")
    logger.info(f"设备: {device}, FP16: {fp16}, vLLM加速: {use_vllm}")
    
    if use_vllm:
        try:
            import vllm
        except ImportError:
            logger.error("启用 vLLM 失败: 未找到 vllm 库。请先安装: pip install vllm==0.9.0")
            sys.exit(1)
    
    start_time = time.time()
    
    try:
        cosy_model = AutoModel(
            model_dir=model_dir,
            load_trt=False,
            load_vllm=use_vllm,
            fp16=fp16
        )
    except TypeError as e:
        if "load_vllm" in str(e):
            logger.error("当前 CosyVoice 版本似乎不支持 vLLM,请确保使用最新代码")
        raise e
    
    logger.info(f"模型加载完成,耗时: {time.time() - start_time:.1f}s")
    logger.info(f"模型采样率: {cosy_model.sample_rate}Hz, 输出采样率: {settings.OUTPUT_SAMPLE_RATE}Hz")
    
    # 初始化音色缓存管理器
    voice_cache_manager = VoiceCacheManager(cosy_model)
    voice_count = voice_cache_manager.load_voices()
    
    # 模型预热
    if settings.ENABLE_MODEL_WARMUP:
        default_voice = voice_cache_manager.get_default_voice()
        if default_voice:
            warmup_model(
                prompt_wav_path=default_voice["file"],
                voice_id=settings.DEFAULT_VOICE_ID
            )
    
    return cosy_model


def warmup_model(prompt_wav_path: str = None, voice_id: str = None):
    """
    预热模型,减少首次请求延迟
    
    Args:
        prompt_wav_path: 参考音频路径
        voice_id: 音色 ID
    """
    global cosy_model
    
    if cosy_model is None:
        logger.warning("模型未加载,跳过预热")
        return
    
    logger.info("🔥 正在预热模型...")
    start_time = time.time()
    
    warmup_text = "预热测试"
    warmup_prompt_text = "You are a helpful assistant.<|endofprompt|>预热"
    
    try:
        if prompt_wav_path and os.path.exists(prompt_wav_path):
            # 使用 zero-shot 模式预热
            spk_id = voice_id if voice_id else "default"
            
            for _ in cosy_model.inference_zero_shot(
                warmup_text,
                warmup_prompt_text,
                prompt_wav_path,
                stream=False,
                zero_shot_spk_id=spk_id
            ):
                pass
            
            logger.info(f"✅ 模型预热完成,耗时: {time.time() - start_time:.1f}s")
        else:
            logger.info("⏭ 跳过预热 (无参考音频)")
            
    except Exception as e:
        logger.warning(f"预热失败 (不影响正常使用): {e}")


def get_cosy_model() -> Optional[AutoModel]:
    """获取全局模型实例"""
    return cosy_model


def get_voice_cache_manager() -> Optional[VoiceCacheManager]:
    """获取音色缓存管理器"""
    return voice_cache_manager


def get_inference_lock() -> threading.Lock:
    """获取推理锁"""
    return inference_lock
