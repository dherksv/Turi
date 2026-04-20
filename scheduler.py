import asyncio
from datetime import datetime
import pytz
import os
from dotenv import load_dotenv
from memory import get_pending_reminders, mark_reminder_fired

load_dotenv()
TIMEZONE = os.getenv("USER_TIMEZONE", "Asia/Kolkata")

# will be set to FastAPI's WebSocket broadcast or SSE queue
# for now just prints — we'll wire to frontend notification later
_notification_callbacks = []

def register_notification(callback):
    _notification_callbacks.append(callback)

async def _fire_reminder(reminder: dict):
    msg = f"⏰ Reminder: {reminder['task']}"
    print(f"\n[SCHEDULER] FIRING → {msg}")
    mark_reminder_fired(reminder["id"])
    for cb in _notification_callbacks:
        await cb(reminder["session_id"], msg)

async def reminder_loop(sse_manager):
    """Background loop — checks reminders every 30 seconds."""
    print("[SCHEDULER] started — checking every 30s")
    tz = pytz.timezone(TIMEZONE)

    while True:
        try:
            now     = datetime.now(tz)
            pending = get_pending_reminders()

            for r in pending:
                remind_at = datetime.fromisoformat(
                    r["remind_at"]
                ).astimezone(tz)

                if now >= remind_at:
                    print(f"[SCHEDULER] firing: {r['task']}")
                    mark_reminder_fired(r["id"])

                    # push to browser via SSE
                    await sse_manager.push_reminder(
                        r["session_id"], r["task"]
                    )

        except Exception as e:
            print(f"[SCHEDULER] error: {e}")

        await asyncio.sleep(30)