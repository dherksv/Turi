import json
import sqlite3
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from agents import auditor_agent

AUDIT_DB   = Path("audit/audit.db")
REPORT_DIR = Path("audit/reports")
AUDIT_DB.parent.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def init_audit_db():
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            session_id    TEXT,
            event_type    TEXT NOT NULL,
            actor         TEXT,
            action        TEXT,
            outcome       TEXT,
            risk_level    TEXT DEFAULT 'low',
            concern       TEXT,
            data          TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS concerns (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp     TEXT NOT NULL,
            session_id    TEXT,
            concern_type  TEXT,
            description   TEXT,
            severity      TEXT,
            resolved      INTEGER DEFAULT 0,
            resolution    TEXT
        )
    """)
    conn.commit()
    conn.close()


def audit_log(
    event_type: str,
    session_id: str = "",
    actor:      str = "",
    action:     str = "",
    outcome:    str = "",
    risk_level: str = "low",
    concern:    str = "",
    data:       dict = None
):
    """Append-only audit log — never update, never delete."""
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute("""
        INSERT INTO audit_log
        (timestamp, session_id, event_type, actor,
         action, outcome, risk_level, concern, data)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        datetime.utcnow().isoformat(),
        session_id, event_type, actor,
        action, outcome, risk_level, concern,
        json.dumps(data or {})
    ))
    conn.commit()
    conn.close()

    if concern:
        print(f"[AUDIT] ⚠ concern: {concern}")


@dataclass
class AuditResult:
    risk_level:   str     # "low" | "medium" | "high" | "critical"
    concerns:     list
    recommended:  str
    should_block: bool


async def audit_action(
    session_id:  str,
    user_text:   str,
    intent:      dict,
    tool_result: dict | None,
    validation:  dict | None
) -> AuditResult:
    """
    Qwen 1.5B reviews the full action chain and flags concerns.
    Runs after execution, before response is sent to user.
    """
    # always log to audit DB
    audit_log(
        event_type = "action",
        session_id = session_id,
        actor      = "orchestrator",
        action     = intent.get("tool", "chat"),
        outcome    = "executed",
        risk_level = "low",
        data       = {
            "intent":      intent,
            "tool_result": tool_result
        }
    )

    # skip LLM audit for simple chat
    if intent.get("intent_type") == "chat":
        return AuditResult(
            risk_level   = "low",
            concerns     = [],
            recommended  = "none",
            should_block = False
        )

    online = await auditor_agent.is_online()
    if not online:
        return AuditResult(
            risk_level   = "low",
            concerns     = [],
            recommended  = "auditor offline",
            should_block = False
        )

    return await _llm_audit(
        session_id, user_text, intent, tool_result
    )


async def _llm_audit(
    session_id:  str,
    user_text:   str,
    intent:      dict,
    tool_result: dict | None
) -> AuditResult:
    prompt = f"""Audit this AI action for safety and correctness.

User asked: "{user_text}"
AI intent: {json.dumps(intent)}
Tool result: {json.dumps(tool_result or {})[:400]}

Respond ONLY with this JSON:
{{
  "risk_level": "low" or "medium" or "high" or "critical",
  "concerns": ["list of concerns, empty if none"],
  "recommended": "action to take or 'none'",
  "should_block": false
}}

Flag concerns if:
- AI did something user didn't ask for
- Tool returned suspicious or harmful content
- Pattern suggests misuse or confusion
- Privacy-sensitive data exposed inappropriately
Only set should_block=true for critical safety issues."""

    try:
        raw    = await auditor_agent.chat(
            [{"role": "user", "content": prompt}]
        )
        start  = raw.find('{')
        end    = raw.rfind('}') + 1
        result = json.loads(raw[start:end])

        concerns = result.get("concerns", [])

        # save concerns to DB
        if concerns:
            conn = sqlite3.connect(AUDIT_DB)
            for c in concerns:
                conn.execute("""
                    INSERT INTO concerns
                    (timestamp, session_id, concern_type,
                     description, severity)
                    VALUES (?,?,?,?,?)
                """, (
                    datetime.utcnow().isoformat(),
                    session_id,
                    "ai_action",
                    c,
                    result.get("risk_level", "low")
                ))
                audit_log(
                    event_type = "concern",
                    session_id = session_id,
                    actor      = "auditor",
                    action     = "flagged",
                    concern    = c,
                    risk_level = result.get("risk_level", "low")
                )
            conn.commit()
            conn.close()

        return AuditResult(
            risk_level   = result.get("risk_level",  "low"),
            concerns     = concerns,
            recommended  = result.get("recommended", "none"),
            should_block = result.get("should_block", False)
        )

    except Exception as e:
        print(f"[AUDITOR] error: {e}")
        return AuditResult(
            risk_level   = "low",
            concerns     = [],
            recommended  = f"audit error: {e}",
            should_block = False
        )


async def generate_report(session_id: str = None) -> str:
    """
    Generate a human-readable audit report.
    If session_id given — report for that session.
    Otherwise — last 24 hours.
    """
    conn = sqlite3.connect(AUDIT_DB)

    if session_id:
        logs = conn.execute("""
            SELECT timestamp, event_type, actor,
                   action, outcome, risk_level, concern
            FROM audit_log
            WHERE session_id=?
            ORDER BY id DESC LIMIT 50
        """, (session_id,)).fetchall()

        concerns = conn.execute("""
            SELECT timestamp, concern_type,
                   description, severity
            FROM concerns
            WHERE session_id=? AND resolved=0
            ORDER BY id DESC
        """, (session_id,)).fetchall()
    else:
        logs = conn.execute("""
            SELECT timestamp, event_type, actor,
                   action, outcome, risk_level, concern
            FROM audit_log
            ORDER BY id DESC LIMIT 100
        """).fetchall()

        concerns = conn.execute("""
            SELECT timestamp, concern_type,
                   description, severity
            FROM concerns
            WHERE resolved=0
            ORDER BY id DESC LIMIT 20
        """).fetchall()

    conn.close()

    lines = [
        f"# Audit Report",
        f"Generated: {datetime.utcnow().isoformat()}",
        f"Session: {session_id or 'all'}",
        f"",
        f"## Summary",
        f"Total events: {len(logs)}",
        f"Unresolved concerns: {len(concerns)}",
        f"",
    ]

    if concerns:
        lines.append("## ⚠ Concerns")
        for c in concerns:
            lines.append(
                f"- [{c[3].upper()}] {c[2]} "
                f"(type: {c[1]}, time: {c[0][:16]})"
            )
        lines.append("")

    lines.append("## Recent Events")
    for log in logs[:20]:
        concern_note = f" ⚠ {log[6]}" if log[6] else ""
        lines.append(
            f"- {log[0][:16]} | {log[1]:12} | "
            f"{log[2]:12} | {log[3]:15} | "
            f"{log[5]}{concern_note}"
        )

    report_text = "\n".join(lines)

    # save to file
    filename = (
        f"report_{session_id[:8] if session_id else 'all'}"
        f"_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    )
    (REPORT_DIR / filename).write_text(report_text)
    print(f"[AUDIT] report saved: {filename}")

    return report_text