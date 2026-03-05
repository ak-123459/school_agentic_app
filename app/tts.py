import io
import threading
import wave
from app.utils.pypipertts import PyPiper
from app.config.config import  TTS_MODELS





# ── Loaded once at import time — no reload on every call ─────────────────────
_piper = PyPiper()
_piper.load_mod()          # pre-load default model (en_US-bryce-medium)
# _sample_rate = _piper.get_sample_rate()
tts_lock = threading.Lock()




def switch_tts_language(lang_code: str):
    global CURRENT_LANGUAGE

    model = TTS_MODELS.get(lang_code, TTS_MODELS["en"])

    if lang_code == CURRENT_LANGUAGE:
        print(f"[TTS] Already using: {model}")
        return

    try:
        _piper.load_mod(model)  # load new voice model
        CURRENT_LANGUAGE = lang_code
        print(f"[TTS] Switched to: {model}")
    except Exception as e:
        print(f"[TTS] Failed to switch model: {e} — keeping current")


def text_to_speech(text: str) -> bytes:
    """Convert text to speech using PyPiper.
    Returns raw WAV bytes — identical interface to the old pyttsx3 version.
    """
    try:
        with tts_lock:
            # Collect all raw PCM chunks from the generator
            pcm_chunks = []
            for chunk in _piper.stream_tts(text):
                pcm_chunks.append(chunk)

            if not pcm_chunks:
                print("[tts] Warning: piper returned no audio")
                return b""

            pcm_data = b"".join(pcm_chunks)

            # Wrap raw PCM in a proper WAV container (in memory, no temp file)
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)       # mono
                wf.setsampwidth(2)       # 16-bit = 2 bytes
                wf.setframerate(22050)
                wf.writeframes(pcm_data)

            return buffer.getvalue()

    except Exception as e:
        print(f"PyPiper TTS Error: {e}")
        return b""






