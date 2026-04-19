from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import asyncio
from scheduler import reminder_loop
from memory import (
    init_db, save_message, get_history,
    get_all_sessions, get_reminders_for_session
)

from memory import init_db, save_message, get_history, get_all_sessions
from llm import chat, health_check
from vector_memory import store_user_fact, store_conversation_summary, retrieve
from normalizer import normalize
from intent import classify
from router import route


# ── models ────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str

class ChatResponse(BaseModel):
    session_id: str
    reply:      str

class MemoryRequest(BaseModel):
    fact: str

class SummaryRequest(BaseModel):
    session_id: str
    summary:    str


app = FastAPI(title="Personal Assistant")
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.on_event("startup")
async def startup():
    init_db()
    # seed a few facts on first run so memory isn't empty
    # comment these out after first run
    store_user_fact("User prefers concise responses without unnecessary padding")
    store_user_fact("User is building a multi-agent AI assistant from scratch")
    store_user_fact("User is based in Thiruvananthapuram, Kerala, India")


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    llm_ok = await health_check()
    return {"api": "ok", "llm": "ok" if llm_ok else "unreachable"}


# ── chat ──────────────────────────────────────────────────────



# replace only the chat_endpoint function:

@app.post("/chat", response_model=ChatResponse)

async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "empty message")

    normalized = normalize(req.message)
    intent     = classify(normalized)

    print(f"\n[CHAT] '{req.message}'")
    print(f"[CHAT] classified → {intent['intent_type']} | tool={intent['tool']} | conf={intent['confidence']}")

    result = await route(intent, req.session_id)
    reply  = result["reply"]

    save_message(req.session_id, "user",      req.message)
    save_message(req.session_id, "assistant", reply)

    # return extra debug info alongside the reply
    return {
        "session_id":  req.session_id,
        "reply":       reply,
        "debug": {
            "intent_type": result.get("intent_type"),
            "tool":        result.get("tool"),
            "routed_to":   result.get("routed_to"),
            "confidence":  result.get("confidence"),
            "tool_status": result.get("tool_result", {}).get("status"),
            "result_count": result.get("tool_result", {}).get("count", 0),
        }
    }


# add this debug endpoint — very useful during development
@app.post("/debug/classify")
async def debug_classify(req: ChatRequest):
    """See exactly how a message gets classified — no LLM call."""
    normalized = normalize(req.message)
    intent     = classify(normalized)
    return intent


# ── memory management endpoints ───────────────────────────────

class MemoryRequest(BaseModel):
    fact: str

@app.post("/memory/fact")
async def add_fact(req: MemoryRequest):
    """Manually store a fact about the user into vector memory."""
    store_user_fact(req.fact)
    return {"stored": True, "fact": req.fact}


class SummaryRequest(BaseModel):
    session_id: str
    summary:    str

@app.post("/memory/summary")
async def add_summary(req: SummaryRequest):
    """Store a conversation summary for long-term memory."""
    store_conversation_summary(req.session_id, req.summary)
    return {"stored": True}


@app.get("/memory/search")
async def search_memory(q: str):
    """Search vector memory — useful for debugging."""
    results = retrieve(q, top_k=5)
    return {"query": q, "results": results}


# ── session endpoints ─────────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    return {"sessions": get_all_sessions()}

@app.get("/history/{session_id}")
async def get_session_history(session_id: str):
    return {"messages": get_history(session_id, limit=50)}

@app.post("/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}

# ── tool endpoints ─────────────────────────────────────────

from tools.search import search_web, results_to_context

@app.get("/search")
async def direct_search(q: str, n: int = 5):
    """Direct search endpoint — test SearXNG without going through chat."""
    result = await search_web(q, num_results=n)
    return result
# ── reminder ─────────────────────────────────────────

# add to startup event
@app.on_event("startup")
async def startup():
    init_db()
    # seed facts — comment out after first run
    store_user_fact("User prefers concise responses")
    store_user_fact("User is building a multi-agent AI assistant")
    store_user_fact("User is based in Thiruvananthapuram Kerala India")
    # start background reminder checker
    asyncio.create_task(reminder_loop())


# add this endpoint alongside existing ones
@app.get("/reminders/{session_id}")
async def list_reminders(session_id: str):
    """See all reminders for a session."""
    return {"reminders": get_reminders_for_session(session_id)}