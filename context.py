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
        print(f"Warning: user_profile.json is not valid JSON — using empty profile")
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


def build_system_prompt(user_message: str) -> str:
    profile = load_profile()

    sections = [
        # identity
        "You are a personal AI assistant. "
        "You are concise, honest, and helpful. "
        "You never make up facts. "
        "If unsure, say so clearly.",

        # runtime
        get_runtime_context(),

        # user profile
        get_profile_context(profile),

        # relevant past memories
        get_memory_context(user_message),

        # behavioral rules
        "\n".join([
            "Rules:",
            "- Address the user by name when natural.",
            "- Keep responses focused and avoid unnecessary padding.",
            "- If the user references something from earlier, use the conversation history.",
            "- Timezone-aware: use the user's local time for anything time-related.",
        ])
    ]

    # filter empty sections and join
    return "\n\n".join(s for s in sections if s.strip())