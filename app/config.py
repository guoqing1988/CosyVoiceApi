import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    MODEL_DIR: str = os.getenv("COSYVOICE_MODEL_DIR", "/data/models/cosyvoice")
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    MAX_WORKERS: int = 4

settings = Settings()
