"""
音色管理服务
提供音色查询、验证、缓存管理等业务逻辑
"""
from typing import Optional, List, Dict
import logging

from ..models import get_voice_cache_manager
from ..schemas import VoiceInfo

logger = logging.getLogger(__name__)


class VoiceService:
    """音色管理服务"""
    
    @staticmethod
    def get_voice_by_id(voice_id: str) -> Optional[Dict]:
        """
        根据 ID 获取音色信息
        
        Args:
            voice_id: 音色 ID
        
        Returns:
            音色信息字典,不存在则返回 None
        """
        manager = get_voice_cache_manager()
        if not manager:
            logger.warning("音色缓存管理器未初始化")
            return None
        
        return manager.get_voice(voice_id)
    
    @staticmethod
    def list_all_voices() -> List[VoiceInfo]:
        """
        获取所有可用音色列表
        
        Returns:
            音色信息列表
        """
        manager = get_voice_cache_manager()
        if not manager:
            logger.warning("音色缓存管理器未初始化")
            return []
        
        voices_data = manager.list_voices()
        return [VoiceInfo(**voice) for voice in voices_data]
    
    @staticmethod
    def get_default_voice() -> Optional[Dict]:
        """
        获取默认音色
        
        Returns:
            默认音色信息
        """
        manager = get_voice_cache_manager()
        if not manager:
            return None
        
        return manager.get_default_voice()
    
    @staticmethod
    def validate_voice_id(voice_id: str) -> bool:
        """
        验证音色 ID 是否存在
        
        Args:
            voice_id: 音色 ID
        
        Returns:
            是否存在
        """
        voice = VoiceService.get_voice_by_id(voice_id)
        return voice is not None
    
    @staticmethod
    def get_voice_count() -> int:
        """
        获取已加载的音色数量
        
        Returns:
            音色数量
        """
        manager = get_voice_cache_manager()
        if not manager:
            return 0
        
        return len(manager.voice_cache)
