"""
LLM Router — decides which model handles each request.

Fast path  → Qwen 1.5B (port 8082)
  - Tool results that just need formatting
  - Simple factual questions
  - Chitchat and acknowledgements
  - Short confirmations
  - Time/date queries

Deep path  → Gemma 4 (port 8081)
  - Reasoning and analysis
  - Suggestions and recommendations
  - Comparisons and trade-offs
  - Creative tasks
  - Multi-step planning
  - Anything needing memory/context synthesis
"""

import re
from agents import chat_agent, fast_agent
from context import build_system_prompt
from debug_logger import log_event

# ── fast path triggers ────────────────────────────────────────

FAST_INTENTS = {
    # tool results — just format and return
    "web_search", "youtube_search", "amazon_search",
    "file_search", "open_file",     "open_app",
    "set_reminder", "play_music",   "open_explorer",
}

FAST_PATTERNS = [
    r"^(what|what's)\s+(the\s+)?(time|date|day)",
    r"^(open|launch|start|run)\s+\w+",
    r"^(play|watch|listen)\s+",
    r"^(remind|set\s+reminder)",
    r"^(yes|no|ok|okay|sure|thanks|thank you|got it|done)",
    r"^(hi|hey|hello|bye|goodbye)",
    r"^(search|find|look\s+up)\s+",
    r"^(buy|shop|order)\s+",
    r"how\s+(much|many)\s+",
    r"^(show|list|display)\s+",
]

# ── deep path triggers ────────────────────────────────────────

DEEP_PATTERNS = [
    r"\b(suggest|recommend|advise|opinion|think|feel|believe)\b",
    r"\b(should\s+i|would\s+you|what\s+do\s+you)\b",
    r"\b(compare|difference|better|best|versus|vs)\b",
    r"\b(explain|describe|analyze|analyse|evaluate)\b",
    r"\b(plan|strategy|approach|method|way\s+to)\b",
    r"\b(help\s+me|how\s+do\s+i|how\s+can\s+i)\b",
    r"\b(create|write|draft|generate|design|build)\b",
    r"\b(why|reason|cause|because|therefore)\b",
    r"\b(problem|issue|error|fix|solve|debug)\b",
    r"\b(future|long.term|sustainable|scalable)\b",
    r"\b(brainstorm|idea|creative|innovative)\b",
    r"\b(pros|cons|advantage|disadvantage|trade.off)\b",
]

DEEP_WORD_THRESHOLD = 12   # long messages → deep path
FAST_WORD_THRESHOLD = 4    # very short → fast path


def should_use_fast_path(
    intent:     dict,
    user_text:  str
) -> tuple[bool, str]:
    """
    Returns (use_fast, reason).
    """
    tool       = intent.get("tool", "")
    text_lower = user_text.lower().strip()
    word_count = len(user_text.split())

    # very short messages — always fast
    if word_count <= FAST_WORD_THRESHOLD:
        return True, "short_message"

    # tool-based fast path
    if tool in FAST_INTENTS:
        return True, f"tool:{tool}"

    # deep pattern check — overrides everything
    for pattern in DEEP_PATTERNS:
        if re.search(pattern, text_lower):
            return False, f"deep_pattern:{pattern[:30]}"

    # fast pattern check
    for pattern in FAST_PATTERNS:
        if re.search(pattern, text_lower):
            return True, f"fast_pattern:{pattern[:30]}"

    # long messages tend to need reasoning
    if word_count > DEEP_WORD_THRESHOLD:
        return False, "long_message"

    # default: fast for commands, deep for questions
    if intent.get("intent_type") == "command":
        return True, "command_default"

    return False, "question_default"


# ── fast path response ────────────────────────────────────────

FAST_SYSTEM = """You are Turi, a quick personal assistant.
Give brief, direct, helpful responses.
For tool results: summarize clearly in 1-3 sentences.
For simple questions: answer directly without preamble.
Never say "certainly" or "of course" or "I'd be happy to".
Be concise. Get to the point immediately."""


async def fast_respond(
    messages:     list[dict],
    user_message: str,
    tool_result:  dict = None,
    input_mode:   str  = "text"
) -> str:
    """Quick response using Qwen 1.5B."""

    online = await fast_agent.is_online()
    if not online:
        # fallback to Gemma if Qwen offline
        print("[LLM ROUTER] fast agent offline — using Gemma")
        return await deep_respond(
            messages, user_message, tool_result, input_mode
        )

    # build lean context for fast agent
    system = FAST_SYSTEM
    if input_mode == "voice":
        system += "\nSpeak naturally, no lists or markdown."

    if tool_result and tool_result.get("status") == "ok":
        import json
        tool_ctx = (
            f"\nTool result: {json.dumps(tool_result)[:800]}"
            f"\nSummarize this for the user."
        )
        system += tool_ctx

    # fast agent only gets last 4 turns — less context = faster
    recent = messages[-4:] if len(messages) > 4 else messages

    log_event("llm_fast_path", "llm_router", {
        "user_message": user_message[:60],
        "tool":         tool_result.get("tool") if tool_result else None
    })

    return await fast_agent.chat(
        recent,
        system = system
    )


# ── deep path response ────────────────────────────────────────

async def deep_respond(
    messages:     list[dict],
    user_message: str,
    tool_result:  dict = None,
    input_mode:   str  = "text"
) -> str:
    """Thoughtful response using Gemma 4."""
    from llm import chat

    log_event("llm_deep_path", "llm_router", {
        "user_message": user_message[:60],
        "reason":       "complex_query"
    })

    return await chat(
        messages     = messages,
        user_message = user_message,
        input_mode   = input_mode
    )


# ── engagement while waiting ──────────────────────────────────

ENGAGEMENT_MESSAGES = {
    "web_search":    "Searching the web for you...",
    "amazon_search": "Checking Amazon for the best options...",
    "youtube_search": "Finding videos for you...",
    "open_file":     "Looking for that file...",
    "open_app":      "Opening that for you...",
    "set_reminder":  "Setting your reminder...",
    "default":       "On it...",
}

def get_engagement_message(tool: str) -> str:
    return ENGAGEMENT_MESSAGES.get(tool, ENGAGEMENT_MESSAGES["default"])