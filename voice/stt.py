import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from faster_whisper import WhisperModel

load_dotenv()
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

# load once at startup — stays in memory
_model = None

def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"[STT] loading whisper {WHISPER_MODEL_SIZE}...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device       = "cpu",
            compute_type = "int8"   # fastest on CPU, good quality
        )
        print(f"[STT] whisper ready")
    return _model


def transcribe(audio_bytes: bytes, extension: str = "webm") -> dict:
    """
    Transcribe audio bytes to text.
    Returns dict with text, language, duration, confidence.
    """
    model = get_model()

    # write to temp file — whisper needs a file path
    with tempfile.NamedTemporaryFile(
        suffix=f".{extension}", delete=False
    ) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size        = 5,
            language         = "en",   # force English — faster
            condition_on_previous_text = False,
            vad_filter       = True,   # skip silence
            vad_parameters   = {
                "min_silence_duration_ms": 500
            }
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        print(f"[STT] transcribed: '{text[:80]}'")
        print(f"[STT] language={info.language} "
              f"duration={info.duration:.1f}s")

        return {
            "text":       text,
            "language":   info.language,
            "duration":   round(info.duration, 2),
            "success":    bool(text),
        }

    except Exception as e:
        print(f"[STT] error: {e}")
        return {
            "text":    "",
            "success": False,
            "error":   str(e)
        }
    finally:
        # clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass