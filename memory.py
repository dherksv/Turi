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