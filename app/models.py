from typing import Optional
import torch
from cosyvoice.cli.cosyvoice import AutoModel

cosy_model = None

def load_cosyvoice_model(model_dir: str):
    global cosy_model
    if cosy_model is None:
        import os
        # Ensure model_dir exists or let AutoModel handle it (snapshot_download)
        cosy_model = AutoModel(
            model_dir=model_dir,
            load_trt=False,
            load_vllm=True,
            fp16=True
        )
    return cosy_model

def get_cosy_model():
    return cosy_model
