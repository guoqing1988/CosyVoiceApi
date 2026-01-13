from pydantic import BaseModel
from typing import Optional, List, Union

class SFTRequest(BaseModel):
    text: str
    speaker: str
    stream: bool = False
    speed: float = 1.0

class ZeroShotRequest(BaseModel):
    text: str
    prompt_text: str
    # prompt_wav will be handled via multipart/form-data if file upload is needed,
    # but for JSON requests we might expect a path or base64.
    # For simplicity in this API, let's assume we can pass a path or we use multipart for the actual file.
    prompt_wav_path: Optional[str] = None
    stream: bool = False
    speed: float = 1.0

class CrossLingualRequest(BaseModel):
    text: str
    prompt_wav_path: str
    stream: bool = False
    speed: float = 1.0

class InstructRequest(BaseModel):
    text: str
    speaker: str
    instruct_text: str
    stream: bool = False
    speed: float = 1.0

class VCRequest(BaseModel):
    source_wav_path: str
    prompt_wav_path: str
    stream: bool = False
    speed: float = 1.0

class TTSRequest(BaseModel):
    text: str
    mode: str = "sft" # sft, zero_shot, cross_lingual, instruct, vc
    speaker: Optional[str] = "中文女"
    prompt_text: Optional[str] = ""
    prompt_wav_path: Optional[str] = ""
    instruct_text: Optional[str] = ""
    source_wav_path: Optional[str] = ""
    stream: bool = False
    speed: float = 1.0
