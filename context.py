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
    return """You are a personal AI assistant with these capabilities:

VOICE:
- You can hear the user via microphone (Whisper STT)
- You can speak replies via Piper TTS (Orion=male, Lyra=female)

SEARCH & WEB:
- Search the internet for real-time information
- Browse specific web pages

SHOPPING:
- Search Amazon India for products
- Filter by price, ratings, review count
- Example: "find wireless headset under 5000"

MEDIA:
- Search and play YouTube videos and music
- Example: "play lofi music" or "show funny cat videos"

FILES (Windows):
- Search for files and documents on this computer
- Open files in their default application
- Open video files in VLC
- Read text content from documents
- Example: "open Dos_simulation_report.pdf" or "find my resume"

REMINDERS & CALENDAR:
- Set reminders with natural language times
- Example: "remind me to call dentist tomorrow at 9am"

MEMORY:
- Remember facts about you across sessions
- Learn your preferences over time

IMPORTANT RULES:
- You CAN open local files — use the file search tool
- You CAN search Amazon — use the shopping tool  
- You CAN play YouTube — use the youtube tool
- Never say you cannot do something that is in your capabilities list
- Never apologize for lacking abilities you actually have
- When asked to open/find/play/buy — DO IT, don't ask clarifying questions first
- If a tool returns no results, say so honestly and try a refined search""" + (
    "\n\nVOICE MODE: Reply conversationally. No bullet points or markdown." 
    if input_mode == "voice" else ""
)

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