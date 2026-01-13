import base64
import soundfile as sf
import io

def wav_to_base64(wav, sample_rate=44100):
    buffer = io.BytesIO()
    sf.write(buffer, wav, sample_rate, format="WAV")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode()

def get_exception_error():
    import traceback
    return traceback.format_exc()