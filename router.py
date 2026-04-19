from tools.registry import dispatch
from llm import chat
from context import build_system_prompt
from memory import get_history, save_message

# confidence below this always goes to LLM for clarification
CONFIDENCE_FLOOR = 0.70

# tools that are safe to execute silently (level 1)
SILENT_TOOLS = {"play_music", "show_info", "open_app", "web_search"}

# tools that need a confirmation reply (level 2)
CONFIRM_TOOLS = {"set_reminder", "create_event"}

# tools that are stubs — tell user honestly
STUB_NOTICE = "This feature isn't connected yet — I've noted what you wanted to do."


async def route(intent: dict, session_id: str) -> dict:
    """
    Takes a classified intent, decides how to handle it.
    Returns a response dict with at minimum a 'reply' key.
    """

    intent_type = intent["intent_type"]
    confidence  = intent["confidence"]
    tool        = intent["tool"]

    # ── low confidence — ask LLM to handle conversationally ──
    if confidence < CONFIDENCE_FLOOR:
        reply = await _llm_response(intent, session_id)
        return {
            "reply":       reply,
            "intent_type": intent_type,
            "routed_to":   "llm_low_confidence",
            "confidence":  confidence
        }

    # ── chitchat — straight to LLM ───────────────────────────
    if intent_type == "chat":
        reply = await _llm_response(intent, session_id)
        return {
            "reply":       reply,
            "intent_type": "chat",
            "routed_to":   "llm",
            "confidence":  confidence
        }

    # ── question — straight to LLM ───────────────────────────
    if intent_type == "question":
        reply = await _llm_response(intent, session_id)
        return {
            "reply":       reply,
            "intent_type": "question",
            "routed_to":   "llm",
            "confidence":  confidence
        }

    # ── command with a tool ───────────────────────────────────
    if intent_type == "command" and tool:

        # call the tool
        tool_result = await dispatch(tool, intent)

        # if tool is a stub — let LLM craft a natural reply
        if tool_result.get("status") == "stub":
            reply = await _llm_with_tool_context(intent, tool_result, session_id)
        else:
            # real tool result — LLM formats it naturally
            reply = await _llm_with_tool_context(intent, tool_result, session_id)

        return {
            "reply":       reply,
            "intent_type": "command",
            "tool":        tool,
            "tool_result": tool_result,
            "routed_to":   "tool",
            "confidence":  confidence
        }

    # ── fallback ──────────────────────────────────────────────
    reply = await _llm_response(intent, session_id)
    return {
        "reply":       reply,
        "intent_type": intent_type,
        "routed_to":   "llm_fallback",
        "confidence":  confidence
    }


async def _llm_response(intent: dict, session_id: str) -> str:
    """Standard LLM call with full context and history."""
    history = get_history(session_id, limit=10)
    history.append({"role": "user", "content": intent["text"]})
    return await chat(messages=history, user_message=intent["text"])


async def _llm_with_tool_context(
    intent:      dict,
    tool_result: dict,
    session_id:  str
) -> str:
    """
    LLM call where we tell the model what the tool returned
    so it can craft a natural language reply.
    """
    history = get_history(session_id, limit=8)

    # inject tool result as context
    tool_context = (
        f"[Tool '{intent['tool']}' was called for: '{intent['text']}']\n"
        f"[Tool result: {tool_result['message']}]\n"
        f"[Note: {tool_result.get('note', '')}]\n\n"
        f"Based on the above, reply naturally to the user. "
        f"If the tool is not connected yet, acknowledge what they asked "
        f"and let them know it's being set up."
    )

    history.append({"role": "user",      "content": intent["text"]})
    history.append({"role": "assistant", "content": tool_context})
    history.append({
        "role":    "user",
        "content": "Now give me your natural reply based on that."
    })

    return await chat(messages=history, user_message=intent["text"])