import os
import re
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from debug_logger import log_event

load_dotenv()
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")

_model = None

# all phonetic variants Whisper might produce for "Turi"
WAKE_WORD_VARIANTS = {
    "turi", "tury", "tori", "tory", "turing",
    "turey", "toori", "touri", "turri", "tuhri",
    "hey turi", "hey tori", "hey tory", "hey turing",
    "hi turi", "hi tori", "ok turi", "okay turi",
    "yo turi", "oi turi","he jury", "Head to read", "head fury", "Hi furi","Hey fury","head","Jury"
}

# what Whisper commonly mishears as wake word
WAKE_WORD_PATTERN = re.compile(
    r'\b(hey\s+)?(tur[iy]|tor[iy]|turing|touri|turri|tuhri)\b',
    re.IGNORECASE
)


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        print(f"[STT] loading whisper {WHISPER_MODEL_SIZE}...")
        _model = WhisperModel(
            WHISPER_MODEL_SIZE,
            device       = "cpu",
            compute_type = "int8"
        )
        print(f"[STT] whisper ready")
    return _model


def transcribe(
    audio_bytes: bytes,
    extension:   str = "webm"
) -> dict:
    model = get_model()

    with tempfile.NamedTemporaryFile(
        suffix=f".{extension}", delete=False
    ) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(
            tmp_path,
            beam_size                   = 5,
            language                    = "en",
            condition_on_previous_text  = False,
            vad_filter                  = True,
            vad_parameters              = {
                "min_silence_duration_ms": 500
            }
        )

        text = " ".join(
            seg.text.strip() for seg in segments
        ).strip()

        # normalize Turi variants in transcript
        text, wake_detected = normalize_wake_word(text)

        log_event("stt_transcribed", "whisper", {
            "text":          text[:100],
            "language":      info.language,
            "duration":      round(info.duration, 2),
            "wake_detected": wake_detected
        })

        print(f"[STT] '{text[:80]}' "
              f"wake={wake_detected}")

        return {
            "text":          text,
            "language":      info.language,
            "duration":      round(info.duration, 2),
            "success":       bool(text),
            "wake_detected": wake_detected
        }

    except Exception as e:
        print(f"[STT] error: {e}")
        return {
            "text":          "",
            "success":       False,
            "error":         str(e),
            "wake_detected": False
        }
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def normalize_wake_word(text: str) -> tuple[str, bool]:
    """
    Replace Whisper's phonetic variants of 'Turi' with 'Turi'.
    Returns (normalized_text, wake_word_detected).
    """
    if WAKE_WORD_PATTERN.search(text):
        normalized = WAKE_WORD_PATTERN.sub(
            lambda m: (
                "Hey Turi" if m.group(1) else "Turi"
            ),
            text
        )
        return normalized, True
    return text, False


def extract_command_after_wake(text: str) -> str:
    """
    Remove the wake word from the start of the text
    to get the actual command.
    'Hey Turi play some music' → 'play some music'
    'Turi what time is it' → 'what time is it'
    """
    cleaned = WAKE_WORD_PATTERN.sub('', text).strip()
    # remove leading punctuation
    cleaned = re.sub(r'^[,.\s]+', '', cleaned).strip()
    return cleaned if cleaned else text