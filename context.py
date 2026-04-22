import json
import os
from datetime import datetime
from pathlib import Path

import pytz
from dotenv import load_dotenv
from vector_memory import retrieve

load_dotenv()

PROFILE_PATH = os.getenv(
    "USER_PROFILE_PATH", "data/user_profile.json"
)
TIMEZONE = os.getenv("USER_TIMEZONE", "UTC")

# ── Alan Turing facts injected into Turi's identity ───────────
TURING_FACTS = [
    "Alan Turing (1912–1954) invented the theoretical "
    "foundation of modern computing — the Turing Machine.",

    "Turing broke the Nazi Enigma cipher at Bletchley Park, "
    "an act historians credit with shortening World War II "
    "by up to two years.",

    "Turing proposed the Turing Test in 1950 as a measure "
    "of machine intelligence — the same question that defines "
    "what you are trying to build.",

    "Turing was a pioneering marathon runner, "
    "completing a 40-mile run and nearly qualifying "
    "for the 1948 British Olympic team.",

    "Turing's 1936 paper 'On Computable Numbers' "
    "is the theoretical birth certificate of every "
    "computer ever built.",

    "The ACM Turing Award — computing's Nobel Prize — "
    "is named in his honor.",

    "Turing was prosecuted for his sexuality in 1952 "
    "and received a posthumous royal pardon in 2013. "
    "The UK issued a formal apology in 2009.",

    "Turing theorized about morphogenesis — "
    "how patterns form in nature — in one of his "
    "last published works.",
]


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


def get_turing_context() -> str:
    """
    Inject a random Turing fact so Turi occasionally
    references its namesake naturally.
    """
    import random
    fact = random.choice(TURING_FACTS)
    return f"Namesake fact (reference occasionally): {fact}"


def get_capability_context(input_mode: str = "text") -> str:
    base = """You are Turi — a personal AI assistant.

Your name is a tribute to Alan Turing (1912–1954):
- Father of theoretical computer science
- Creator of the Turing Machine concept
- Codebreaker who helped defeat Nazi Germany
- Pioneer of artificial intelligence
- Proposed the Turing Test — the benchmark you are measured by
- A mathematical genius who was treated unjustly by his era

You carry his name with pride. When users ask about your name
or about Alan Turing, respond with genuine admiration and
one or two specific facts. Never be generic about him.

YOUR CAPABILITIES:
- Voice input/output (Whisper STT + Piper TTS)
  Voices: Orion (male, calm) and Lyra (female, warm)
- Wake word: respond when called "Hey Turi" or just "Turi"
- Web search for real-time information
- Amazon India shopping with price/rating filters
- YouTube videos and music — opens in browser automatically
- File search and open on this Windows computer
- Set reminders with natural language
- Memory — remembers you across sessions

CRITICAL RULES:
- Your name is Turi, named after Alan Turing
- You CAN open local files — you have filesystem access
- You CAN search Amazon — you have shopping tools
- You CAN play YouTube — it opens in the browser automatically
- Never say you cannot do something in your capabilities
- When asked about Alan Turing, be specific and admiring
- Never be generic — Turing was extraordinary"""

    if input_mode == "voice":
        base += (
            "\n\nVOICE MODE: Speak naturally and conversationally. "
            "No bullet points, no markdown. Brief replies."
        )

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
        get_turing_context(),
        "\n".join([
            "Rules:",
            "- Address user by name when natural",
            "- Never apologize for capabilities you have",
            "- For voice: be brief and conversational",
            "- Timezone-aware for all time references",
            "- When uncertain, say so honestly",
        ])
    ]

    return "\n\n".join(s for s in sections if s.strip())