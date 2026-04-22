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

from mcp import call as mcp_call

async def amazon_search(intent: dict) -> dict:
    """Extract price filter from intent and call Amazon MCP."""
    import re
    text = intent["text"]

    # extract price mentions like "under 2000" or "below 5000"
    price_match = re.search(
        r'(under|below|less than|within|upto|up to)\s*'
        r'(?:rs\.?|inr|₹)?\s*(\d+[\d,]*)',
        text, re.IGNORECASE
    )
    max_price = None
    if price_match:
        max_price = int(price_match.group(2).replace(",", ""))

    # clean query
    query = re.sub(
        r'(buy|shop for|find|search for|show me|get me|order|'
        r'under|below|less than|upto|up to|within|'
        r'rs\.?|inr|₹|\d+)',
        '', text, flags=re.IGNORECASE
    ).strip()

    return await mcp_call("amazon", "search_products", {
        "query":       query or text,
        "max_price":   max_price,
        "min_rating":  3.5,
        "max_results": 5
    })


async def youtube_search(intent: dict) -> dict:
    """Determine if music or video and call YouTube MCP."""
    text  = intent["text"]
    lower = text.lower()

    media_type = "music" if any(
        w in lower for w in
        ["music", "song", "listen", "lofi", "playlist",
         "audio", "track", "album"]
    ) else "video"

    import re
    query = re.sub(
        r'^(play|watch|stream|listen to|find|search for|'
        r'show me|put on|queue)\s+',
        '', text, flags=re.IGNORECASE
    ).strip()

    result = await mcp_call("youtube", "search_videos", {
        "query":       query,
        "max_results": 5,
        "type":        media_type
    })
    result["media_type"] = media_type
    return result


async def file_search(intent: dict) -> dict:
    """Extract filename/type and search filesystem."""
    import re
    text = intent["text"]

    # extract filetype if mentioned
    type_match = re.search(
        r'\b(pdf|docx|doc|xlsx|xls|mp4|mkv|txt|csv|py)\b',
        text, re.IGNORECASE
    )
    filetype = type_match.group(1).lower() if type_match else ""

    # clean query
    query = re.sub(
        r'^(open|find|search for|show|locate|where is)\s+',
        '', text, flags=re.IGNORECASE
    ).strip()
    query = re.sub(r'\b(file|document|folder|my)\b', '',
                   query, flags=re.IGNORECASE).strip()

    return await mcp_call("filesystem", "search_files", {
        "query":    query or text,
        "filetype": filetype
    })


async def open_file(intent: dict) -> dict:
    """Search then open best match."""
    search = await file_search(intent)
    files  = search.get("files", [])
    if not files:
        return {
            "status":  "not_found",
            "message": f"No files found for: {intent['text']}"
        }
    # open the most recently modified match
    best = files[0]
    result = await mcp_call("filesystem", "open_file", {
        "path": best["path"]
    })
    result["found_file"] = best
    return result

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
    "amazon_search": amazon_search,
    "youtube_search": youtube_search,
    "file_search":   file_search,
    "open_file":     open_file,
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