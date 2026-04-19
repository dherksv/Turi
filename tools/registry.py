"""
Tool registry — each tool is a function that takes an intent
dict and returns a result dict.

These are stubs for now. Replace the body of each function
as you build each real tool.
"""

async def set_reminder(intent: dict) -> dict:
    entities = intent.get("entities", {})
    time_refs = entities.get("time_refs", [])
    return {
        "status":  "stub",
        "message": f"[set_reminder] would schedule: '{intent['text']}'",
        "time_refs": time_refs,
        "note":    "reminder tool not yet connected"
    }


async def web_search(intent: dict) -> dict:
    return {
        "status":  "stub",
        "message": f"[web_search] would search for: '{intent['text']}'",
        "note":    "search tool not yet connected"
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