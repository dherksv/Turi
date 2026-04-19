"""
Tool registry — each tool is a function that takes an intent
dict and returns a result dict.

These are stubs for now. Replace the body of each function
as you build each real tool.
"""
import re
import os
from dotenv import load_dotenv
from tools.search import search_web, results_to_context
from tools.reminder import build_reminder_intent

load_dotenv()


async def set_reminder(intent: dict) -> dict:
    """Parse reminder details — actual saving happens after user confirms."""
    parsed = build_reminder_intent(intent)

    if not parsed["has_time"]:
        return {
            "status":   "needs_time",
            "task":     parsed["task"],
            "message":  f"I understood you want to be reminded to: "
                        f"'{parsed['task']}' — but when? "
                        f"Please tell me a time or day."
        }

    from tools.reminder import format_remind_at
    human_time = format_remind_at(parsed["remind_at"])

    return {
        "status":    "ready_to_confirm",
        "task":      parsed["task"],
        "remind_at": parsed["remind_at"],
        "human_time": human_time,
        "message":   f"Set a reminder to '{parsed['task']}' "
                     f"on {human_time}?"
    }

async def web_search(intent: dict) -> dict:
    """Real web search via SearXNG."""
    query = intent["text"]

    # strip command verb prefix
    query = re.sub(
        r'^(search(\s+about)?\s+|find\s+|look\s+up\s+|google\s+|browse\s+)',
        '', query, flags=re.IGNORECASE
    ).strip()

    if not query:
        query = intent["text"]

    result = await search_web(query, num_results=5)
    context_str = results_to_context(result)

    return {
        "status":      result["status"],
        "query":       query,
        "context_str": context_str,
        "count":       result.get("count", 0),
        "raw_results": result.get("results", []),
        "message":     context_str
    }


async def play_music(intent: dict) -> dict:
    return {
        "status":  "stub",
        "message": f"[play_music] would play: '{intent['text']}'",
        "note":    "music tool not yet connected"
    }


async def send_message(intent: dict) -> dict:
    contact = intent.get("entities", {}).get("contact", "unknown")
    return {
        "status":  "stub",
        "message": f"[send_message] would message {contact}: '{intent['text']}'",
        "note":    "messaging tool not yet connected"
    }


async def make_call(intent: dict) -> dict:
    contact = intent.get("entities", {}).get("contact", "unknown")
    return {
        "status":  "stub",
        "message": f"[make_call] would call {contact}",
        "note":    "call tool not yet connected"
    }


async def create_event(intent: dict) -> dict:
    return {
        "status":  "stub",
        "message": f"[create_event] would create: '{intent['text']}'",
        "note":    "calendar tool not yet connected"
    }


async def show_info(intent: dict) -> dict:
    return {
        "status":  "stub",
        "message": f"[show_info] would display: '{intent['text']}'",
        "note":    "info tool not yet connected"
    }


async def open_app(intent: dict) -> dict:
    return {
        "status":  "stub",
        "message": f"[open_app] would open: '{intent['text']}'",
        "note":    "app launcher not yet connected"
    }


# ── tool dispatch map ─────────────────────────────────────────

TOOL_MAP = {
    "set_reminder":  set_reminder,
    "web_search":    web_search,
    "play_music":    play_music,
    "send_message":  send_message,
    "make_call":     make_call,
    "create_event":  create_event,
    "show_info":     show_info,
    "open_app":      open_app,
}


async def dispatch(tool_name: str, intent: dict) -> dict:
    """Call the right tool by name. Returns error dict if unknown."""
    fn = TOOL_MAP.get(tool_name)
    if not fn:
        return {
            "status":  "error",
            "message": f"unknown tool: {tool_name}"
        }
    return await fn(intent)