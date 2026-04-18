from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid

from memory import init_db, save_message, get_history, get_all_sessions
from llm import chat, health_check
from vector_memory import store_user_fact, store_conversation_summary, retrieve

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

class ChatRequest(BaseModel):
    session_id: str
    message:    str

class ChatResponse(BaseModel):
    session_id: str
    reply:      str

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "empty message")

    # load history
    history = get_history(req.session_id, limit=10)

    # append current message to history for LLM
    history.append({"role": "user", "content": req.message})

    # call LLM — now passes user_message for memory retrieval
    reply = await chat(
        messages     = history,
        user_message = req.message
    )

    # persist both turns
    save_message(req.session_id, "user",      req.message)
    save_message(req.session_id, "assistant", reply)

    return ChatResponse(session_id=req.session_id, reply=reply)


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