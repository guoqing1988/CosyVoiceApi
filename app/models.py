from typing import Optional
import torch
import sys
import time
from cosyvoice.cli.cosyvoice import AutoModel
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cosy_model = None

def load_cosyvoice_model(model_dir: str, device: str = "cuda", fp16: bool = True, use_vllm: bool = True):
    global cosy_model
    if cosy_model is None:
        logger.info(f"正在加载模型: {model_dir}")
        logger.info(f"设备: {device}, FP16: {fp16}, vLLM加速: {use_vllm}")

        if use_vllm:
            try:
                import vllm
            except ImportError:
                logger.error("启用 vLLM 失败: 未找到 vllm 库。请先安装: pip install vllm==0.9.0")
                sys.exit(1)
        # Ensure model_dir exists or let AutoModel handle it (snapshot_download)
        start_time = time.time()
        cosy_model = AutoModel(
            model_dir=model_dir,
            load_trt=False,
            load_vllm=use_vllm,
            fp16=fp16
        )
        logger.info(f"模型加载完成，耗时: {time.time() - start_time:.1f}s")
        logger.info(f"模型采样率: {cosy_model.sample_rate}Hz")
    return cosy_model

def get_cosy_model():
    return cosy_model
