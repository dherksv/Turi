"""
Debug Logger — structured JSON logging for all agent activity.
Writes to logs/debug.jsonl (one JSON per line, easy to parse)
and logs/debug.log (human readable)
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

DEBUG_JSONL  = LOG_DIR / "debug.jsonl"
DEBUG_LOG    = LOG_DIR / "debug.log"
ERROR_LOG    = LOG_DIR / "errors.log"

# human readable logger
logging.basicConfig(
    level   = logging.DEBUG,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [
        logging.FileHandler(DEBUG_LOG),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("assistant")


def log_event(
    event:      str,
    agent:      str,
    data:       dict = None,
    error:      str  = None,
    session_id: str  = "",
    task_id:    str  = "",
    duration_ms: int = 0
):
    """Write a structured JSON log entry."""
    entry = {
        "ts":         datetime.utcnow().isoformat(),
        "event":      event,
        "agent":      agent,
        "session_id": session_id[:8] if session_id else "",
        "task_id":    task_id[:8]    if task_id    else "",
        "duration_ms": duration_ms,
        "data":       data  or {},
        "error":      error or None
    }

    # write to JSONL
    with open(DEBUG_JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # write to human log
    if error:
        logger.error(
            f"[{agent}] {event} | {error} | {json.dumps(data or {})[:200]}"
        )
        # also write to error log
        with open(ERROR_LOG, "a") as f:
            f.write(json.dumps({**entry, "traceback": error}) + "\n")
    else:
        logger.debug(
            f"[{agent}] {event} | {json.dumps(data or {})[:200]}"
        )

    return entry


def log_handoff(
    from_agent:  str,
    to_agent:    str,
    payload:     dict,
    session_id:  str = "",
    task_id:     str = ""
):
    """Log agent-to-agent handoff."""
    log_event(
        event      = "handoff",
        agent      = from_agent,
        data       = {
            "from":    from_agent,
            "to":      to_agent,
            "payload": _safe_truncate(payload)
        },
        session_id = session_id,
        task_id    = task_id
    )
    logger.info(
        f"[HANDOFF] {from_agent} → {to_agent} | "
        f"{json.dumps(_safe_truncate(payload))[:150]}"
    )


def log_tool_call(
    tool:       str,
    args:       dict,
    result:     dict,
    agent:      str     = "executor",
    session_id: str     = "",
    task_id:    str     = "",
    duration_ms: int    = 0,
    error:      str     = None
):
    """Log a tool call with full input/output."""
    log_event(
        event      = "tool_call",
        agent      = agent,
        data       = {
            "tool":   tool,
            "args":   _safe_truncate(args),
            "result": _safe_truncate(result),
            "status": result.get("status", "unknown")
                      if result else "error"
        },
        error      = error,
        session_id = session_id,
        task_id    = task_id,
        duration_ms = duration_ms
    )


def log_classification(
    text:    str,
    intent:  dict,
    session_id: str = ""
):
    """Log intent classification result."""
    log_event(
        event      = "classification",
        agent      = "classifier",
        data       = {
            "input":       text[:100],
            "intent_type": intent.get("intent_type"),
            "tool":        intent.get("tool"),
            "confidence":  intent.get("confidence"),
            "entities":    intent.get("entities", {})
        },
        session_id = session_id
    )


def log_agent_response(
    agent:      str,
    input_data: Any,
    output:     Any,
    session_id: str = "",
    task_id:    str = "",
    duration_ms: int = 0
):
    """Log any agent's input/output."""
    log_event(
        event      = "agent_response",
        agent      = agent,
        data       = {
            "input":  _safe_truncate(input_data),
            "output": _safe_truncate(output)
        },
        session_id = session_id,
        task_id    = task_id,
        duration_ms = duration_ms
    )


def log_error(
    agent:      str,
    error:      Exception,
    context:    dict = None,
    session_id: str  = "",
    task_id:    str  = ""
):
    """Log an exception with full traceback."""
    tb = traceback.format_exc()
    log_event(
        event      = "error",
        agent      = agent,
        data       = context or {},
        error      = f"{type(error).__name__}: {str(error)}\n{tb}",
        session_id = session_id,
        task_id    = task_id
    )


def _safe_truncate(data: Any, max_len: int = 500) -> Any:
    """Truncate long values for logging."""
    if isinstance(data, dict):
        return {
            k: _safe_truncate(v, max_len)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [_safe_truncate(i, max_len) for i in data[:5]]
    if isinstance(data, str) and len(data) > max_len:
        return data[:max_len] + "...[truncated]"
    return data


def get_recent_logs(limit: int = 50) -> list[dict]:
    """Read recent log entries from JSONL file."""
    if not DEBUG_JSONL.exists():
        return []
    lines = DEBUG_JSONL.read_text().strip().split("\n")
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return list(reversed(entries))


def get_error_logs(limit: int = 20) -> list[dict]:
    """Read recent error entries."""
    if not ERROR_LOG.exists():
        return []
    lines = ERROR_LOG.read_text().strip().split("\n")
    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:
            continue
    return list(reversed(entries))