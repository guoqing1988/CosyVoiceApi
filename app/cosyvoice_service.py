import numpy as np

def text_to_speech(model, text: str, **kwargs):
    """
    单次合成：文本 -> 语音
    """
    waveform = model.infer(text, **kwargs)  # CosyVoice TTS 输出
    return waveform

def stream_tts(model, text: str):
    """
    示例流式返回（按分片发送）
    """
    size = 4096
    wav = model.infer(text)
    for offset in range(0, len(wav), size):
        yield wav[offset:offset + size]
