import json
import os
from datetime import datetime
from pathlib import Path

import pytz
from dotenv import load_dotenv
from vector_memory import retrieve

load_dotenv()

PROFILE_PATH = os.getenv("USER_PROFILE_PATH", "data/user_profile.json")
TIMEZONE     = os.getenv("USER_TIMEZONE", "UTC")


def load_profile() -> dict:
    path = Path(PROFILE_PATH)
    if not path.exists():
        return {}
    content = path.read_text().strip()
    if not content:
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def get_runtime_context() -> str:
    tz  = pytz.timezone(TIMEZONE)
    now = datetime.now(tz)
    return "\n".join([
        f"Current date : {now.strftime('%A, %d %B %Y')}",
        f"Current time : {now.strftime('%I:%M %p')} ({TIMEZONE})",
    ])


def get_profile_context(profile: dict) -> str:
    if not profile:
        return ""
    lines = ["User profile:"]
    for key, val in profile.items():
        if isinstance(val, list):
            lines.append(f"  {key}: {', '.join(val)}")
        else:
            lines.append(f"  {key}: {val}")
    return "\n".join(lines)


def get_memory_context(user_message: str) -> str:
    memories = retrieve(user_message, top_k=3)
    if not memories:
        return ""
    lines = ["Relevant memories:"]
    for mem in memories:
        lines.append(f"  - {mem}")
    return "\n".join(lines)


def get_capability_context(input_mode: str = "text") -> str:
    """
    Tell the LLM exactly what it is and what it can do.
    Changes tone based on whether input came from voice or text.
    """
    base = """You are a personal AI assistant with the following capabilities:
- Text chat: read and reply to typed messages
- Voice input: you can hear the user speak via microphone (Whisper STT)
- Voice output: you can speak replies aloud via Piper TTS
  - Male voice: Orion — deep, calm, focused
  - Female voice: Lyra — clear, warm, expressive
- Web search: find real-time information from the internet
- Reminders: set and manage timed reminders
- Memory: remember facts about the user across sessions"""

    if input_mode == "voice":
        base += """

The user is currently speaking to you via voice.
Your reply will be spoken aloud by your voice system.
Keep replies concise and conversational — avoid bullet points,
markdown, long lists, or any formatting that sounds unnatural
when spoken. Speak naturally as if in conversation."""
    else:
        base += """

The user is currently typing to them.
You may use markdown formatting in replies where helpful."""

    return base


def build_system_prompt(
    user_message: str,
    input_mode:   str = "text"
) -> str:
    profile = load_profile()

    sections = [
        get_capability_context(input_mode),
        get_runtime_context(),
        get_profile_context(profile),
        get_memory_context(user_message),
        "\n".join([
            "Rules:",
            "- Address the user by name when natural.",
            "- Never say you cannot speak — you have a voice system.",
            "- Never say you are text-only — you are multimodal.",
            "- For voice replies: be brief, natural, conversational.",
            "- For text replies: be clear and well-formatted.",
            "- Never make up facts. If unsure say so.",
            "- Timezone-aware: use the user's local time.",
        ])
    ]

    return "\n\n".join(s for s in sections if s.strip())