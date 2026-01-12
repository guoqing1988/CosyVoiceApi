from vllm import LLMEngine, SamplingParams
import asyncio
from .config import settings

engine = None

def init_vllm(model_path: str):
    global engine
    if engine is None:
        engine = LLMEngine.from_model(
            model=model_path,
            dtype="torch.bfloat16",
            device="cuda"
        )
    return engine

async def stream_tokens(prompt: str):
    params = SamplingParams(max_tokens=500, use_beam_search=False)
    async for token in engine.stream_generate(prompt=prompt, sampling_params=params):
        yield token
