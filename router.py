from tools.registry import dispatch
from llm import chat, stream_chat
from memory import get_history, save_reminder

CONFIDENCE_FLOOR = 0.70
_pending_confirmations: dict = {}


async def route(
    intent:     dict,
    session_id: str,
    input_mode: str = "text"    # ← new param
) -> dict:

    intent_type = intent["intent_type"]
    confidence  = intent["confidence"]
    tool        = intent["tool"]
    text        = intent["text"]

    print(f"\n[ROUTER] type={intent_type} | tool={tool} | "
          f"conf={confidence} | mode={input_mode}")

    # confirmation check
    confirmation = _check_confirmation(text, session_id)
    if confirmation:
        return await _handle_confirmation(confirmation, session_id)

    if confidence < CONFIDENCE_FLOOR:
        reply = await _llm_response(intent, session_id, input_mode)
        return {"reply": reply, "intent_type": intent_type,
                "routed_to": "llm_low_confidence",
                "confidence": confidence}

    if intent_type in ("chat", "question"):
        reply = await _llm_response(intent, session_id, input_mode)
        return {"reply": reply, "intent_type": intent_type,
                "routed_to": "llm", "confidence": confidence}

    if intent_type == "command" and tool:
        tool_result = await dispatch(tool, intent)

        if tool == "web_search":
            reply = await _llm_with_search(
                intent, tool_result, session_id, input_mode
            )
        elif tool == "set_reminder":
            reply = await _handle_reminder(
                tool_result, intent, session_id
            )
        elif tool_result.get("status") == "stub":
            reply = await _llm_with_stub(
                intent, tool_result, session_id, input_mode
            )

        elif tool == "amazon_search":
            reply = await _format_amazon(
                tool_result, intent, session_id, input_mode
            )
        elif tool == "youtube_search":
            reply = await _format_youtube(
                tool_result, intent, session_id, input_mode
            )
        elif tool in ("file_search", "open_file"):
            reply = await _format_files(
                tool_result, intent, session_id, input_mode
            )    
        else:
            reply = await _llm_with_tool_context(
                intent, tool_result, session_id, input_mode
            )

        return {
            "reply":       reply,
            "intent_type": "command",
            "tool":        tool,
            "tool_result": tool_result,
            "routed_to":   "tool",
            "confidence":  confidence
        }

    reply = await _llm_response(intent, session_id, input_mode)
    return {"reply": reply, "intent_type": intent_type,
            "routed_to": "llm_fallback", "confidence": confidence}


# ── reminder handlers (unchanged) ────────────────────────────

async def _handle_reminder(
    tool_result: dict,
    intent:      dict,
    session_id:  str
) -> str:
    status = tool_result.get("status")
    if status == "needs_time":
        return tool_result["message"]
    if status == "ready_to_confirm":
        _pending_confirmations[session_id] = {
            "type":       "reminder",
            "task":       tool_result["task"],
            "remind_at":  tool_result["remind_at"],
            "human_time": tool_result["human_time"]
        }
        return (
            f"Got it. I'll remind you to {tool_result['task']} "
            f"on {tool_result['human_time']}. Confirm? yes or no"
        )
    return "I couldn't parse that reminder. Could you rephrase?"


def _check_confirmation(text: str, session_id: str) -> dict | None:
    if session_id not in _pending_confirmations:
        return None
    lower    = text.lower().strip()
    yes_words = {"yes","yeah","yep","sure","ok","okay",
                 "confirm","do it","go ahead","set it"}
    no_words  = {"no","nope","cancel","don't","stop","nevermind"}
    if any(w in lower for w in yes_words):
        pending = _pending_confirmations.pop(session_id)
        pending["confirmed"] = True
        return pending
    if any(w in lower for w in no_words):
        _pending_confirmations.pop(session_id)
        return {"confirmed": False, "type": "reminder"}
    return None


async def _handle_confirmation(
    confirmation: dict,
    session_id:   str
) -> dict:
    if not confirmation["confirmed"]:
        return {"reply": "Okay, reminder cancelled.",
                "routed_to": "hitl_cancelled"}
    reminder_id = save_reminder(
        session_id = session_id,
        task       = confirmation["task"],
        remind_at  = confirmation["remind_at"]
    )
    return {
        "reply": (
            f"Done. Reminder set. "
            f"I'll remind you to {confirmation['task']} "
            f"on {confirmation['human_time']}."
        ),
        "routed_to":   "reminder_saved",
        "reminder_id": reminder_id
    }


# ── LLM helpers — all accept input_mode ──────────────────────

async def _llm_response(
    intent:     dict,
    session_id: str,
    input_mode: str = "text"
) -> str:
    history = get_history(session_id, limit=10)
    history.append({"role": "user", "content": intent["text"]})
    return await chat(
        messages     = history,
        user_message = intent["text"],
        input_mode   = input_mode
    )


async def _llm_with_search(
    intent:      dict,
    tool_result: dict,
    session_id:  str,
    input_mode:  str = "text"
) -> str:
    history = get_history(session_id, limit=6)
    status  = tool_result.get("status", "error")
    query   = tool_result.get("query", intent["text"])
    count   = tool_result.get("count", 0)
    error   = tool_result.get("error", "unknown")

    if status == "error":
        system_addition = (
            f"Search failed for '{query}' ({error}). "
            f"Answer from your own knowledge. "
            f"Mention live search was unavailable."
        )
    elif count == 0:
        system_addition = (
            f"Search for '{query}' returned no results. "
            f"Answer from your own knowledge."
        )
    else:
        voice_note = (
            " Keep your answer brief and spoken-word natural."
            if input_mode == "voice" else ""
        )
        system_addition = (
            f"{tool_result.get('context_str', '')}\n\n"
            f"Using the search results above answer: "
            f"'{intent['text']}'\n"
            f"Be concise. Cite sources naturally.{voice_note}"
        )

    messages = (
        history
        + [{"role": "system", "content": system_addition}]
        + [{"role": "user",   "content": intent["text"]}]
    )
    return await chat(
        messages     = messages,
        user_message = intent["text"],
        input_mode   = input_mode
    )


async def _llm_with_stub(
    intent:      dict,
    tool_result: dict,
    session_id:  str,
    input_mode:  str = "text"
) -> str:
    history = get_history(session_id, limit=8)
    note = (
        f"User asked: '{intent['text']}'. "
        f"The '{intent['tool']}' feature is not connected yet. "
        f"Acknowledge what they want and say it's being set up. Be brief."
    )
    history.append({"role": "system", "content": note})
    history.append({"role": "user",   "content": intent["text"]})
    return await chat(
        messages     = history,
        user_message = intent["text"],
        input_mode   = input_mode
    )


async def _llm_with_tool_context(
    intent:      dict,
    tool_result: dict,
    session_id:  str,
    input_mode:  str = "text"
) -> str:
    history = get_history(session_id, limit=8)
    context = (
        f"Tool '{intent['tool']}' returned: "
        f"{tool_result.get('message', '')}\n"
        f"Reply naturally to the user."
    )
    history.append({"role": "system", "content": context})
    history.append({"role": "user",   "content": intent["text"]})
    return await chat(
        messages     = history,
        user_message = intent["text"],
        input_mode   = input_mode
    )

async def _format_amazon(
    result: dict, intent: dict,
    session_id: str, input_mode: str
) -> str:
    if result.get("status") == "error":
        return f"Couldn't search Amazon right now: {result.get('error')}"

    products = result.get("products", [])
    if not products:
        return "No products found matching your criteria on Amazon."

    # build context for LLM
    lines = [f"Amazon search results for '{result.get('query')}':\n"]
    for i, p in enumerate(products, 1):
        lines.append(
            f"{i}. {p['title']}\n"
            f"   Price: {p['price_str']} | "
            f"Rating: {p['rating']}⭐ ({p['review_count']:,} reviews)\n"
            f"   Link: {p['link']}\n"
        )

    context = "\n".join(lines)
    history = get_history(session_id, limit=4)
    voice_note = " Summarize top 2-3 picks conversationally." \
        if input_mode == "voice" else ""

    messages = (
        history
        + [{"role": "system", "content":
            f"{context}\n\nPresent these results helpfully. "
            f"Highlight best value. Mention rating and review count. "
            f"Suggest the top pick.{voice_note}"}]
        + [{"role": "user", "content": intent["text"]}]
    )
    return await chat(
        messages=messages,
        user_message=intent["text"],
        input_mode=input_mode
    )


async def _format_youtube(
    result: dict, intent: dict,
    session_id: str, input_mode: str
) -> str:
    if result.get("status") == "error":
        return f"YouTube search failed: {result.get('error')}"

    videos = result.get("videos", [])
    if not videos:
        return "No videos found on YouTube."

    media_type = result.get("media_type", "video")
    lines = [f"YouTube {media_type} results:\n"]
    for i, v in enumerate(videos, 1):
        dur = f"{v['duration']//60}:{v['duration']%60:02d}" \
            if v.get("duration") else "N/A"
        lines.append(
            f"{i}. {v['title']}\n"
            f"   Channel: {v['uploader']} | "
            f"Duration: {dur} | "
            f"Views: {v.get('view_count',0):,}\n"
            f"   URL: {v['url']}\n"
        )

    context = "\n".join(lines)
    history = get_history(session_id, limit=4)

    messages = (
        history
        + [{"role": "system", "content":
            f"{context}\n\nRecommend the best option. "
            f"Explain briefly why it's a good pick. "
            f"Include the YouTube URL so user can click it."}]
        + [{"role": "user", "content": intent["text"]}]
    )
    return await chat(
        messages=messages,
        user_message=intent["text"],
        input_mode=input_mode
    )


async def _format_files(
    result: dict, intent: dict,
    session_id: str, input_mode: str
) -> str:
    status = result.get("status")

    if status == "not_found":
        return result.get("message", "No files found.")

    if status == "ok" and result.get("found_file"):
        # file was opened
        f = result["found_file"]
        return (
            f"Found and opened **{f['name']}** "
            f"({f['size_kb']} KB)\n"
            f"Path: {f['path']}\n"
            f"{result.get('message', '')}"
        )

    files = result.get("files", [])
    if not files:
        return "No matching files found."

    lines = ["Found these files:\n"]
    for i, f in enumerate(files, 1):
        lines.append(
            f"{i}. {f['name']} "
            f"({f['size_kb']} KB) — {f['path']}"
        )
    lines.append(
        "\nSay 'open [filename]' to open any of these."
    )
    return "\n".join(lines)