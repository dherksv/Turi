import sqlite3
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()
DB_PATH = Path(os.getenv("DB_PATH", "data/chat.db"))
DB_PATH.parent.mkdir(exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT    NOT NULL,
            role        TEXT    NOT NULL,
            content     TEXT    NOT NULL,
            timestamp   TEXT    NOT NULL
        )
    """)
     # new — reminders table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS reminders (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id   TEXT    NOT NULL,
            task         TEXT    NOT NULL,
            remind_at    TEXT    NOT NULL,
            created_at   TEXT    NOT NULL,
            fired        INTEGER NOT NULL DEFAULT 0,
            confirmed    INTEGER NOT NULL DEFAULT 0
        )
    """)

    conn.commit()
    conn.close()

def save_message(session_id: str, role: str, content: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?,?,?,?)",
        (session_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_history(session_id: str, limit: int = 10) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT role, content FROM messages
           WHERE session_id=?
           ORDER BY id DESC LIMIT ?""",
        (session_id, limit)
    ).fetchall()
    conn.close()
    # reverse so oldest first
    return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

def get_all_sessions() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT DISTINCT session_id FROM messages ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]

def save_reminder(
    session_id: str,
    task:       str,
    remind_at:  str   # ISO format datetime string
) -> int:
    """Save a confirmed reminder. Returns its id."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        """INSERT INTO reminders
           (session_id, task, remind_at, created_at, fired, confirmed)
           VALUES (?,?,?,?,0,1)""",
        (session_id, task, remind_at,
         datetime.utcnow().isoformat())
    )
    reminder_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return reminder_id


def get_pending_reminders() -> list[dict]:
    """Get all confirmed, unfired reminders."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT id, session_id, task, remind_at
           FROM reminders
           WHERE fired=0 AND confirmed=1
           ORDER BY remind_at ASC"""
    ).fetchall()
    conn.close()
    return [
        {
            "id":         r[0],
            "session_id": r[1],
            "task":       r[2],
            "remind_at":  r[3]
        }
        for r in rows
    ]


def mark_reminder_fired(reminder_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "UPDATE reminders SET fired=1 WHERE id=?",
        (reminder_id,)
    )
    conn.commit()
    conn.close()


def get_reminders_for_session(session_id: str) -> list[dict]:
    """List all reminders for a session — fired and pending."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT id, task, remind_at, fired
           FROM reminders
           WHERE session_id=?
           ORDER BY remind_at ASC""",
        (session_id,)
    ).fetchall()
    conn.close()
    return [
        {
            "id":        r[0],
            "task":      r[1],
            "remind_at": r[2],
            "fired":     bool(r[3])
        }
        for r in rows
    ]