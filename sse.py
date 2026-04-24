import asyncio
import json
from datetime import datetime


class SSEManager:
    """
    Manages SSE connections per session.
    When a reminder fires, we push it to the right browser tab.
    """

    def __init__(self):
        # session_id → list of queues (one per open tab)
        self._queues: dict[str, list[asyncio.Queue]] = {}

    def subscribe(self, session_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        if session_id not in self._queues:
            self._queues[session_id] = []
        self._queues[session_id].append(q)
        print(f"[SSE] session {session_id[:8]} subscribed "
              f"({len(self._queues[session_id])} listeners)")
        return q

    def unsubscribe(self, session_id: str, q: asyncio.Queue):
        if session_id in self._queues:
            self._queues[session_id].discard(q) \
                if hasattr(self._queues[session_id], 'discard') \
                else None
            try:
                self._queues[session_id].remove(q)
            except ValueError:
                pass
        print(f"[SSE] session {session_id[:8]} unsubscribed")

    async def push(
        self,
        session_id:  str,
        event_type:  str,
        data:        dict
    ):
        """Push an event to all listeners for a session."""
        if session_id not in self._queues:
            print(f"[SSE] no listeners for session "
                  f"{session_id[:8]} — event '{event_type}' dropped")
            return

        payload = json.dumps({
            "type":      event_type,
            "data":      data,
            "timestamp": datetime.utcnow().isoformat()
        })

        listeners = self._queues[session_id]
        print(f"[SSE] pushing '{event_type}' to "
              f"session {session_id[:8]} "
              f"({len(listeners)} listeners)")

        dead = []
        for q in listeners:
            try:
                await q.put(payload)
                print(f"[SSE] delivered '{event_type}' to queue")
            except Exception as e:
                print(f"[SSE] queue put failed: {e}")
                dead.append(q)

        for q in dead:
            try:
                listeners.remove(q)
            except ValueError:
                pass
            
    async def push_reminder(self, session_id: str, task: str):
        await self.push(session_id, "reminder", {"task": task})

    async def push_notification(
        self, session_id: str, message: str, level: str = "info"
    ):
        await self.push(session_id, "notification",
                        {"message": message, "level": level})


# global singleton — imported everywhere
sse_manager = SSEManager()