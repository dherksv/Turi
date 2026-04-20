import os
import subprocess
import tempfile
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PIPER_BINARY    = os.getenv("PIPER_BINARY",    r"C:\piper\piper.exe")
VOICE_MODELS_DIR = Path(os.getenv("VOICE_MODELS_DIR", "voice/models"))
DEFAULT_VOICE   = os.getenv("DEFAULT_VOICE",   "orion")

VOICES = {
    "orion": {
        "name":        "Orion",
        "gender":      "male",
        "model":       VOICE_MODELS_DIR / "orion.onnx",
        "config":      VOICE_MODELS_DIR / "orion.onnx.json",
        "description": "Deep, calm, focused"
    },
    "lyra": {
        "name":        "Lyra",
        "gender":      "female",
        "model":       VOICE_MODELS_DIR / "lyra.onnx",
        "config":      VOICE_MODELS_DIR / "lyra.onnx.json",
        "description": "Clear, warm, expressive"
    }
}


def _clean_text_for_speech(text: str) -> str:
    """
    Remove markdown and formatting that sounds bad when spoken.
    **bold** → bold, # Header → Header, etc.
    """
    # remove markdown bold/italic
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    # remove markdown headers
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # remove URLs
    text = re.sub(r'https?://\S+', 'link', text)
    # remove bullet points
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # remove code blocks
    text = re.sub(r'```[\s\S]*?```', 'code block', text)
    text = re.sub(r'`(.+?)`', r'\1', text)
    # collapse whitespace
    text = re.sub(r'\n{2,}', '. ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def synthesize(
    text:       str,
    voice_name: str = None,
    speed:      float = 1.0
) -> bytes | None:
    """
    Convert text to speech using Piper.
    Returns WAV audio bytes or None on failure.
    """
    voice_key = (voice_name or DEFAULT_VOICE).lower()
    voice     = VOICES.get(voice_key, VOICES[DEFAULT_VOICE])

    model_path = voice["model"]
    if not model_path.exists():
        print(f"[TTS] model not found: {model_path}")
        print(f"[TTS] download it to voice/models/")
        return None

    cleaned_text = _clean_text_for_speech(text)
    if not cleaned_text:
        return None

    print(f"[TTS] {voice['name']} speaking: '{cleaned_text[:60]}...'")

    # write output to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            PIPER_BINARY,
            "--model",        str(model_path),
            "--config",       str(voice["config"]),
            "--output_file",  tmp_path,
            "--length_scale", str(1.0 / speed),  # speed up = shorter
        ]

        result = subprocess.run(
            cmd,
            input        = cleaned_text.encode("utf-8"),
            capture_output = True,
            timeout       = 30
        )

        if result.returncode != 0:
            print(f"[TTS] piper error: {result.stderr.decode()}")
            return None

        audio_bytes = Path(tmp_path).read_bytes()
        print(f"[TTS] generated {len(audio_bytes)} bytes")
        return audio_bytes

    except subprocess.TimeoutExpired:
        print(f"[TTS] piper timed out")
        return None
    except FileNotFoundError:
        print(f"[TTS] piper binary not found at: {PIPER_BINARY}")
        print(f"[TTS] check PIPER_BINARY in .env")
        return None
    except Exception as e:
        print(f"[TTS] error: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def list_voices() -> list[dict]:
    result = []
    for key, v in VOICES.items():
        result.append({
            "key":         key,
            "name":        v["name"],
            "gender":      v["gender"],
            "description": v["description"],
            "available":   v["model"].exists()
        })
    return result