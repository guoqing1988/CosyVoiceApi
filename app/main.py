from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import os
import logging

from .config import settings
from .models import load_cosyvoice_model
from .utils import get_exception_error
from .controllers import system, voice, tts

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CosyVoice API",
    description="高并发语音生成 API 服务",
    version="2.0.0"
)

# 注册路由
app.include_router(system.router)
app.include_router(voice.router)
app.include_router(tts.router)

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


# ========== 静态文件服务 ==========

asset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "asset")
if os.path.exists(asset_dir):
    app.mount("/asset", StaticFiles(directory=asset_dir), name="asset")
    logger.info(f"📁 asset文件目录: {asset_dir}")


# 挂载静态文件 - 必须放在最后以避免遮蔽 API 路由
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    logger.info(f"📁 静态文件目录: {static_dir}")