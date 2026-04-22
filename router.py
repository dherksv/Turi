from tools.registry import dispatch
from llm import chat, stream_chat
from memory import get_history, save_reminder
import uuid
from pipeline import (
    validate_intent,
    execute_with_recovery,
    audit_action,
    audit_log
)

CONFIDENCE_FLOOR = 0.70
_pending_confirmations: dict = {}


async def route(
    intent:     dict,
    session_id: str,
    input_mode: str = "text"
) -> dict:

    intent_type = intent["intent_type"]
    confidence  = intent["confidence"]
    tool        = intent["tool"]
    text        = intent["text"]

    print(f"\n[ROUTER] type={intent_type} | tool={tool} | "
          f"conf={confidence:.2f} | mode={input_mode}")

    # ── step 1: check pending confirmations ──────────────────
    confirmation = _check_confirmation(text, session_id)
    if confirmation:
        return await _handle_confirmation(confirmation, session_id)

    # ── step 2: validate intent ───────────────────────────────
    validation = await validate_intent(
        user_text = text,
        intent    = intent,
        use_llm   = (intent_type == "command"
                     and confidence > 0.7)
    )

    print(f"[VALIDATOR] verdict={validation.verdict} "
          f"reason={validation.reason}")

    if not validation.passed:
        audit_log(
            event_type = "validation_rejected",
            session_id = session_id,
            actor      = "validator",
            action     = tool or intent_type,
            outcome    = validation.verdict,
            risk_level = "medium",
            concern    = validation.reason
        )
        if validation.verdict == "reject":
            return {
                "reply": (
                    f"I want to make sure I understood you. "
                    f"Could you rephrase that? "
                    f"({validation.reason})"
                ),
                "intent_type": intent_type,
                "routed_to":   "validation_rejected"
            }
        # fix — use corrected intent
        if validation.fixed:
            intent = {**intent, **validation.fixed}
            tool   = intent.get("tool")

    # ── step 3: low confidence — LLM handles it ──────────────
    if confidence < CONFIDENCE_FLOOR:
        reply = await _llm_response(intent, session_id, input_mode)
        return {"reply": reply, "intent_type": intent_type,
                "routed_to": "llm_low_confidence",
                "confidence": confidence}

    # ── step 4: chat and questions ────────────────────────────
    if intent_type in ("chat", "question"):
        reply = await _llm_response(intent, session_id, input_mode)
        audit_log(
            event_type = "chat",
            session_id = session_id,
            actor      = "orchestrator",
            action     = intent_type,
            outcome    = "ok"
        )
        return {"reply": reply, "intent_type": intent_type,
                "routed_to": "llm", "confidence": confidence}

    # ── step 5: commands — execute with recovery ──────────────
    if intent_type == "command" and tool:
        task_id = str(uuid.uuid4())

        async def execute():
            return await dispatch(tool, intent)

        recovery_result = await execute_with_recovery(
            task_id    = task_id,
            session_id = session_id,
            intent     = intent,
            execute_fn = execute,
            max_attempts = 3
        )

        if recovery_result["status"] in ("failed", "exhausted"):
            audit_log(
                event_type = "tool_failed",
                session_id = session_id,
                actor      = "executor",
                action     = tool,
                outcome    = "failed",
                risk_level = "high",
                concern    = recovery_result.get("error", "")
            )
            return {
                "reply": (
                    f"I tried to {tool} but ran into an issue: "
                    f"{recovery_result.get('error', 'unknown error')}. "
                    f"Please try again or rephrase."
                ),
                "routed_to": "tool_failed"
            }

        tool_result = recovery_result.get("result", {})

        # ── step 6: audit the action ──────────────────────────
        audit = await audit_action(
            session_id  = session_id,
            user_text   = text,
            intent      = intent,
            tool_result = tool_result,
            validation  = {"verdict": validation.verdict,
                           "reason":  validation.reason}
        )

        if audit.should_block:
            audit_log(
                event_type = "blocked_by_audit",
                session_id = session_id,
                actor      = "auditor",
                action     = tool,
                outcome    = "blocked",
                risk_level = "critical",
                concern    = str(audit.concerns)
            )
            return {
                "reply": (
                    "I'm not able to complete that action — "
                    "it was flagged by the safety audit. "
                    f"Concerns: {', '.join(audit.concerns)}"
                ),
                "routed_to": "audit_blocked"
            }

        # ── step 7: format reply ──────────────────────────────
        if tool == "web_search":
            reply = await _llm_with_search(
                intent, tool_result, session_id, input_mode
            )
        elif tool == "set_reminder":
            reply = await _handle_reminder(
                tool_result, intent, session_id
            )
        elif tool in ("amazon_search",):
            reply = await _format_amazon(
                tool_result, intent, session_id, input_mode
            )
        elif tool in ("youtube_search",):
            reply = await _format_youtube(
                tool_result, intent, session_id, input_mode
            )
        elif tool in ("file_search", "open_file"):
            reply = await _format_files(
                tool_result, intent, session_id, input_mode
            )
        elif tool_result.get("status") == "stub":
            reply = await _llm_with_stub(
                intent, tool_result, session_id, input_mode
            )
        else:
            reply = await _llm_with_tool_context(
                intent, tool_result, session_id, input_mode
            )

        audit_log(
            event_type = "tool_success",
            session_id = session_id,
            actor      = "executor",
            action     = tool,
            outcome    = "ok",
            risk_level = audit.risk_level
        )

        return {
            "reply":       reply,
            "intent_type": "command",
            "tool":        tool,
            "tool_result": tool_result,
            "routed_to":   "tool",
            "confidence":  confidence,
            "audit":       {
                "risk_level": audit.risk_level,
                "concerns":   audit.concerns
            }
        }

    # fallback
    reply = await _llm_response(intent, session_id, input_mode)
    return {"reply": reply, "intent_type": intent_type,
            "routed_to": "llm_fallback",
            "confidence": confidence}

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