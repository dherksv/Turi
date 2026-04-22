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
from pipeline.agent_memory import task_memory_factory

CONFIDENCE_FLOOR = 0.70
_pending_confirmations: dict = {}


import uuid
from pipeline.agent_memory import task_memory_factory

async def route(
    intent:     dict,
    session_id: str,
    input_mode: str = "text"
) -> dict:

    # create isolated memory for this task
    task_id  = str(uuid.uuid4())
    memories = task_memory_factory.create_task_memory(
        task_id    = task_id,
        session_id = session_id
    )

    # each agent gets its own context
    orch_mem  = memories["orchestrator"]
    valid_mem = memories["validator"]
    audit_mem = memories["auditor"]

    intent_type = intent["intent_type"]
    confidence  = intent["confidence"]
    tool        = intent["tool"]
    text        = intent["text"]

    print(f"\n[ROUTER] task={task_id[:8]} "
          f"type={intent_type} tool={tool}")

    # ── store user request in orchestrator working memory ─────
    orch_mem.working.set("user_text",   text)
    orch_mem.working.set("intent",      intent)
    orch_mem.working.set("session_id",  session_id)
    orch_mem.working.set("started_at",  datetime.utcnow().isoformat())

    # ── confirmation check ────────────────────────────────────
    confirmation = _check_confirmation(text, session_id)
    if confirmation:
        result = await _handle_confirmation(confirmation, session_id)
        task_memory_factory.cleanup_task(task_id)
        return result

    # ── validation ────────────────────────────────────────────
    validation = await validate_intent(
        user_text = text,
        intent    = intent,
        use_llm   = (intent_type == "command"
                     and confidence > 0.7)
    )

    # validator writes its result to scratchpad
    await valid_mem.write_result(
        content      = {
            "verdict":    validation.verdict,
            "reason":     validation.reason,
            "confidence": validation.confidence
        },
        content_type = "validation_result"
    )

    if not validation.passed:
        task_memory_factory.cleanup_task(task_id)
        if validation.verdict == "reject":
            return {
                "reply": (
                    f"Could you rephrase that? "
                    f"({validation.reason})"
                ),
                "routed_to": "validation_rejected"
            }
        if validation.fixed:
            intent = {**intent, **validation.fixed}
            tool   = intent.get("tool")

    # ── low confidence / chat / question ─────────────────────
    if (confidence < CONFIDENCE_FLOOR
            or intent_type in ("chat", "question")):
        reply = await _llm_response(intent, session_id, input_mode)
        # orchestrator stores reply in working memory
        orch_mem.working.set("reply", reply)
        task_memory_factory.cleanup_task(task_id)
        return {
            "reply":       reply,
            "intent_type": intent_type,
            "routed_to":   "llm",
            "confidence":  confidence
        }

    # ── command execution ─────────────────────────────────────
    if intent_type == "command" and tool:

        async def execute():
            return await dispatch(tool, intent)

        recovery_result = await execute_with_recovery(
            task_id      = task_id,
            session_id   = session_id,
            intent       = intent,
            execute_fn   = execute,
            max_attempts = 3
        )

        if recovery_result["status"] in ("failed", "exhausted"):
            task_memory_factory.cleanup_task(task_id)
            return {
                "reply": (
                    f"I ran into an issue with {tool}: "
                    f"{recovery_result.get('error', 'unknown')}. "
                    f"Please try again."
                ),
                "routed_to": "tool_failed"
            }

        tool_result = recovery_result.get("result", {})

        # save tool context to conversation so next turn remembers
        if tool_result.get("status") == "ok":
            context_summary = _summarize_tool_result(
                tool, tool_result
            )
            if context_summary:
                save_message(
                    session_id, "system",
                    f"[Tool result: {context_summary}]"
                )
        # tool agent writes result to scratchpad

        tool_agent_name = f"{tool}_agent"
        if tool_agent_name in memories:
            tool_mem = memories[tool_agent_name]
        else:
            tool_mem = memories["orchestrator"]

        await tool_mem.write_result(
            content      = tool_result,
            content_type = f"{tool}_result"
        )

        # ── audit ─────────────────────────────────────────────
        audit = await audit_action(
            session_id  = session_id,
            user_text   = text,
            intent      = intent,
            tool_result = tool_result,
            validation  = {
                "verdict": validation.verdict,
                "reason":  validation.reason
            }
        )

        # auditor writes concerns to scratchpad
        if audit.concerns:
            await audit_mem.write_result(
                content      = {
                    "concerns":   audit.concerns,
                    "risk_level": audit.risk_level
                },
                content_type = "audit_concerns"
            )

        if audit.should_block:
            task_memory_factory.cleanup_task(task_id)
            return {
                "reply": (
                    "That action was blocked by the safety audit. "
                    f"Concerns: {', '.join(audit.concerns)}"
                ),
                "routed_to": "audit_blocked"
            }

        # ── format reply ──────────────────────────────────────
        reply = await _format_reply(
            tool, tool_result, intent, session_id, input_mode
        )

        # orchestrator stores final reply in working memory
        orch_mem.working.set("reply",       reply)
        orch_mem.working.set("tool_result", tool_result)

        # orchestrator decides if anything worth promoting
        # to long-term memory
        if _worth_promoting(tool, tool_result):
            summary = f"User used {tool}: {text[:100]}"
            await orch_mem.promote_to_long_term(
                content      = summary,
                content_type = "task_summary"
            )

        # cleanup scratchpad after task
        task_memory_factory.cleanup_task(task_id)

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

    reply = await _llm_response(intent, session_id, input_mode)
    task_memory_factory.cleanup_task(task_id)
    return {
        "reply":      reply,
        "intent_type": intent_type,
        "routed_to":  "llm_fallback"
    }


def _worth_promoting(tool: str, result: dict) -> bool:
    """Decide if a tool result is worth long-term storage."""
    if result.get("status") != "ok":
        return False
    # only promote meaningful interactions
    return tool in {
        "set_reminder", "create_event",
        "web_search", "amazon_search"
    }


async def _format_reply(
    tool:        str,
    tool_result: dict,
    intent:      dict,
    session_id:  str,
    input_mode:  str
) -> str:
    if tool == "web_search":
        return await _llm_with_search(
            intent, tool_result, session_id, input_mode
        )
    elif tool == "set_reminder":
        return await _handle_reminder(
            tool_result, intent, session_id
        )
    elif tool == "amazon_search":
        return await _format_amazon(
            tool_result, intent, session_id, input_mode
        )
    elif tool == "youtube_search":
        return await _format_youtube(
            tool_result, intent, session_id, input_mode
        )
    elif tool in ("file_search", "open_file"):
        return await _format_files(
            tool_result, intent, session_id, input_mode
        )
    elif tool_result.get("status") == "stub":
        return await _llm_with_stub(
            intent, tool_result, session_id, input_mode
        )
    return await _llm_with_tool_context(
        intent, tool_result, session_id, input_mode
    )
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
    result:     dict,
    intent:     dict,
    session_id: str,
    input_mode: str
) -> str:
    if result.get("status") == "error":
        error = result.get("error", "unknown")
        # give helpful error
        if "yt-dlp" in error:
            return ("YouTube search isn't available right now. "
                    "Make sure yt-dlp is installed: "
                    "`pip install yt-dlp`")
        return f"YouTube search failed: {error}"

    videos = result.get("videos", [])
    if not videos:
        return (f"I searched YouTube for '{result.get('query')}' "
                f"but found no results. Try different keywords.")

    media_type = result.get("type", result.get("media_type", "video"))
    lines      = [f"YouTube results for '{result.get('query')}':\n"]

    for i, v in enumerate(videos, 1):
        dur = ""
        if v.get("duration"):
            m   = v["duration"] // 60
            s   = v["duration"] % 60
            dur = f"{m}:{s:02d}"
        lines.append(
            f"{i}. {v['title']}\n"
            f"   Channel: {v.get('uploader','')}"
            f"{' | ' + dur if dur else ''}\n"
            f"   URL: {v['url']}\n"
        )

    context  = "\n".join(lines)
    history  = get_history(session_id, limit=6)
    voice_note = (
        " Give a brief spoken recommendation of the top pick."
        if input_mode == "voice" else ""
    )

    messages = (
        history
        + [{
            "role":    "system",
            "content": (
                f"{context}\n\n"
                f"The user asked: '{intent['text']}'\n"
                f"You found {len(videos)} YouTube results above.\n"
                f"Present the top 2-3 results with their URLs "
                f"so the user can click them. "
                f"Be enthusiastic and helpful.{voice_note}"
            )
        }]
        + [{"role": "user", "content": intent["text"]}]
    )

    return await chat(
        messages     = messages,
        user_message = intent["text"],
        input_mode   = input_mode
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

def _summarize_tool_result(tool: str, result: dict) -> str:
    """Short summary of tool result saved to conversation history."""
    try:
        if tool == "amazon_search":
            products = result.get("products", [])
            if products:
                names = [p["title"][:40] for p in products[:3]]
                return (f"Amazon search found {len(products)} products: "
                        f"{', '.join(names)}")

        if tool == "youtube_search":
            videos = result.get("videos", [])
            if videos:
                titles = [v["title"][:40] for v in videos[:3]]
                return (f"YouTube found {len(videos)} videos: "
                        f"{', '.join(titles)}")

        if tool in ("file_search", "open_file"):
            files = result.get("files", [])
            if files:
                names = [f["name"] for f in files[:3]]
                return f"Found files: {', '.join(names)}"

        if tool == "web_search":
            count = result.get("count", 0)
            query = result.get("query", "")
            return f"Web search for '{query}' returned {count} results"

    except Exception:
        pass
    return ""