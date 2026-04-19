from tools.registry import dispatch
from llm import chat
from memory import get_history, save_reminder

CONFIDENCE_FLOOR = 0.70

# in-memory pending confirmations
# key: session_id → pending reminder dict
_pending_confirmations: dict = {}


async def route(intent: dict, session_id: str) -> dict:
    intent_type = intent["intent_type"]
    confidence  = intent["confidence"]
    tool        = intent["tool"]
    text        = intent["text"]

    print(f"\n[ROUTER] type={intent_type} | tool={tool} | conf={confidence}")
    print(f"[ROUTER] text='{text}'")

    # ── check if this message is a confirmation reply ─────────
    confirmation = _check_confirmation(text, session_id)
    if confirmation:
        return await _handle_confirmation(confirmation, session_id)

    if confidence < CONFIDENCE_FLOOR:
        reply = await _llm_response(intent, session_id)
        return {"reply": reply, "intent_type": intent_type,
                "routed_to": "llm_low_confidence", "confidence": confidence}

    if intent_type in ("chat", "question"):
        reply = await _llm_response(intent, session_id)
        return {"reply": reply, "intent_type": intent_type,
                "routed_to": "llm", "confidence": confidence}

    if intent_type == "command" and tool:
        print(f"[ROUTER] → dispatching tool: {tool}")
        tool_result = await dispatch(tool, intent)
        print(f"[ROUTER] tool result status: {tool_result.get('status')}")

        if tool == "web_search":
            reply = await _llm_with_search(intent, tool_result, session_id)

        elif tool == "set_reminder":
            reply = await _handle_reminder(
                tool_result, intent, session_id
            )

        elif tool_result.get("status") == "stub":
            reply = await _llm_with_stub(intent, tool_result, session_id)

        else:
            reply = await _llm_with_tool_context(
                intent, tool_result, session_id
            )

        return {
            "reply":       reply,
            "intent_type": "command",
            "tool":        tool,
            "tool_result": tool_result,
            "routed_to":   "tool",
            "confidence":  confidence
        }

    reply = await _llm_response(intent, session_id)
    return {"reply": reply, "intent_type": intent_type,
            "routed_to": "llm_fallback", "confidence": confidence}


# ── reminder HITL ─────────────────────────────────────────────

async def _handle_reminder(
    tool_result: dict,
    intent:      dict,
    session_id:  str
) -> str:
    status = tool_result.get("status")

    # no time found — ask for it
    if status == "needs_time":
        return tool_result["message"]

    # ready — store as pending and ask user to confirm
    if status == "ready_to_confirm":
        _pending_confirmations[session_id] = {
            "type":      "reminder",
            "task":      tool_result["task"],
            "remind_at": tool_result["remind_at"],
            "human_time": tool_result["human_time"]
        }
        print(f"[ROUTER] reminder pending confirmation for session {session_id}")
        return (
            f"Got it. I'll remind you to **{tool_result['task']}** "
            f"on **{tool_result['human_time']}**.\n\n"
            f"Confirm? (yes / no)"
        )

    return "I couldn't parse that reminder. Could you rephrase it?"


def _check_confirmation(text: str, session_id: str) -> dict | None:
    """Check if this message is a yes/no reply to a pending confirmation."""
    if session_id not in _pending_confirmations:
        return None

    lower = text.lower().strip()
    yes_words = {"yes", "yeah", "yep", "sure", "ok",
                 "okay", "confirm", "do it", "go ahead", "set it"}
    no_words  = {"no", "nope", "cancel", "don't", "stop", "nevermind"}

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
        return {
            "reply":     "Okay, reminder cancelled.",
            "routed_to": "hitl_cancelled"
        }

    # save to SQLite
    reminder_id = save_reminder(
        session_id = session_id,
        task       = confirmation["task"],
        remind_at  = confirmation["remind_at"]
    )

    print(f"[REMINDER] saved id={reminder_id} "
          f"task='{confirmation['task']}' "
          f"at={confirmation['remind_at']}")

    return {
        "reply": (
            f"Done! Reminder set ✓\n"
            f"I'll remind you to **{confirmation['task']}** "
            f"on **{confirmation['human_time']}**."
        ),
        "routed_to":   "reminder_saved",
        "reminder_id": reminder_id
    }


# ── standard LLM handlers ─────────────────────────────────────

async def _llm_response(intent: dict, session_id: str) -> str:
    history = get_history(session_id, limit=10)
    history.append({"role": "user", "content": intent["text"]})
    return await chat(messages=history, user_message=intent["text"])


async def _llm_with_search(
    intent:      dict,
    tool_result: dict,
    session_id:  str
) -> str:
    history = get_history(session_id, limit=6)
    status  = tool_result.get("status", "error")
    query   = tool_result.get("query", intent["text"])
    count   = tool_result.get("count", 0)
    error   = tool_result.get("error", "unknown error")

    if status == "error":
        system_addition = (
            f"Search failed for '{query}' ({error}). "
            f"Answer from your own knowledge, "
            f"mention live search was unavailable."
        )
    elif count == 0:
        system_addition = (
            f"Search for '{query}' returned no results. "
            f"Answer from your own knowledge."
        )
    else:
        system_addition = (
            f"{tool_result.get('context_str', '')}\n\n"
            f"Using the search results above, answer: '{intent['text']}'\n"
            f"Be concise. Cite sources naturally. "
            f"Do not invent anything not in the results."
        )

    messages = (
        history
        + [{"role": "system", "content": system_addition}]
        + [{"role": "user",   "content": intent["text"]}]
    )
    return await chat(messages=messages, user_message=intent["text"])


async def _llm_with_stub(
    intent:      dict,
    tool_result: dict,
    session_id:  str
) -> str:
    history = get_history(session_id, limit=8)
    note = (
        f"User asked: '{intent['text']}'. "
        f"The '{intent['tool']}' feature is not connected yet. "
        f"Acknowledge what they want, confirm you understood, "
        f"say this feature is being set up. Be brief."
    )
    history.append({"role": "system", "content": note})
    history.append({"role": "user",   "content": intent["text"]})
    return await chat(messages=history, user_message=intent["text"])


async def _llm_with_tool_context(
    intent:      dict,
    tool_result: dict,
    session_id:  str
) -> str:
    history = get_history(session_id, limit=8)
    context = (
        f"Tool '{intent['tool']}' returned: "
        f"{tool_result.get('message', '')}\n"
        f"Reply naturally to the user."
    )
    history.append({"role": "system", "content": context})
    history.append({"role": "user",   "content": intent["text"]})
    return await chat(messages=history, user_message=intent["text"])