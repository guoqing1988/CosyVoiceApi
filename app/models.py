from typing import Optional
import torch

cosy_model = None

def load_cosyvoice_model(model_dir: str):
    global cosy_model
    if cosy_model is None:
        from cosyvoice.cli.cosyvoice import AutoModel
        cosy_model = AutoModel(
            model_dir=model_dir,
            load_trt=False, 
            load_vllm=True,
            fp16=True
        )
    return cosy_model
