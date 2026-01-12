from pydantic import BaseModel

class TTSRequest(BaseModel):
    text: str

class TTSResponse(BaseModel):
    filename: str
    sample_rate: int = 44100
