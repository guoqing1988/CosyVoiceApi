import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_DIR: str = os.getenv("COSYVOICE_MODEL_DIR", "/data/models/cosyvoice")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_WORKERS: int = 4
    # VOICE_CONFIGS : List[VoiceConfig] = [
    #     {
    #         "id": "default",  # 默认音色
    #         "file": "zero_shot_prompt.wav",  # asset/zero_shot_prompt.wav
    #         "prompt_text": "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
    #     },
    #     # 添加更多音色示例 (取消注释并修改):
    #     {
    #         "id": "tone_man",
    #         "file": "tone_man.wav",
    #         "prompt_text": "You are a helpful assistant.<|endofprompt|>真不好意思，从小至今，他还从来没有被哪一位异性朋友亲吻过呢。"
    #     },
    #     {
    #         "id": "tone_woman",
    #         "file": "tone_woman.wav", 
    #         "prompt_text": "You are a helpful assistant.<|endofprompt|>我们将为全球城市的可持续发展贡献力量。"
    #     },
    #     {
    #         "id": "tone_woman2",
    #         "file": "tone_woman2.wav",
    #         "prompt_text": "You are a helpful assistant.<|endofprompt|>您好，我是智能电话助手，很高兴为您服务。请问您需要咨询业务预约办理还是查询信息？"
    #     }
    # ]

settings = Settings()
