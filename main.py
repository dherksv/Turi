import asyncio
import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import uuid
from fastapi import UploadFile, File, Form
from fastapi.responses import Response
from voice.stt import transcribe
from voice.tts import synthesize, list_voices

from memory import (
    init_db, save_message, get_history,
    get_all_sessions, get_reminders_for_session
)
from llm import chat, health_check
from vector_memory import store_user_fact, store_conversation_summary, retrieve
from normalizer import normalize
from intent import classify
from router import route
from tools.search import search_web
from sse import sse_manager
from scheduler import reminder_loop

app = FastAPI(title="Personal Assistant")
app.mount("/static", StaticFiles(directory="frontend"), name="static")


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


@app.on_event("startup")
async def startup():
    init_db()
    store_user_fact("User prefers concise responses")
    store_user_fact("User is building a multi-agent AI assistant")
    store_user_fact("User is based in Thiruvananthapuram Kerala India")
    # pass sse_manager to scheduler
    asyncio.create_task(reminder_loop(sse_manager))


@app.get("/")
async def root():
    return FileResponse("frontend/index.html")


@app.get("/health")
async def health():
    agents = await health_check()
    return {"api": "ok", **agents}


# ── SSE endpoint ──────────────────────────────────────────────

@app.get("/events/{session_id}")
async def sse_endpoint(session_id: str, request: Request):
    """
    Browser connects here to receive real-time events
    (reminders, notifications) for this session.
    """
    q = sse_manager.subscribe(session_id)

    async def event_stream():
        try:
            # send connected confirmation
            yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

            while True:
                # check if client disconnected
                if await request.is_disconnected():
                    break

                try:
                    # wait for next event with timeout
                    payload = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {payload}\n\n"
                except asyncio.TimeoutError:
                    # send keepalive ping
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        except asyncio.CancelledError:
            pass
        finally:
            sse_manager.unsubscribe(session_id, q)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "Access-Control-Allow-Origin": "*",
        }
    )


# ── chat ──────────────────────────────────────────────────────

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "empty message")

    normalized = normalize(req.message)
    intent     = classify(normalized)

    print(f"\n[CHAT] '{req.message}'")
    print(f"[CHAT] → {intent['intent_type']} | "
          f"tool={intent['tool']} | conf={intent['confidence']}")

    result = await route(intent, req.session_id)
    reply  = result["reply"]

    save_message(req.session_id, "user",      req.message)
    save_message(req.session_id, "assistant", reply)

    return {
        "session_id": req.session_id,
        "reply":      reply,
        "debug": {
            "intent_type":  result.get("intent_type"),
            "tool":         result.get("tool"),
            "routed_to":    result.get("routed_to"),
            "confidence":   result.get("confidence"),
            "tool_status":  result.get("tool_result", {}).get("status"),
            "result_count": result.get("tool_result", {}).get("count", 0),
        }
    }


# ── memory ────────────────────────────────────────────────────

@app.post("/memory/fact")
async def add_fact(req: MemoryRequest):
    store_user_fact(req.fact)
    return {"stored": True, "fact": req.fact}

@app.post("/memory/summary")
async def add_summary(req: SummaryRequest):
    store_conversation_summary(req.session_id, req.summary)
    return {"stored": True}

@app.get("/memory/search")
async def search_memory(q: str):
    results = retrieve(q, top_k=5)
    return {"query": q, "results": results}


# ── sessions + reminders ──────────────────────────────────────

@app.get("/sessions")
async def list_sessions():
    return {"sessions": get_all_sessions()}

@app.get("/history/{session_id}")
async def get_session_history(session_id: str):
    return {"messages": get_history(session_id, limit=50)}

@app.post("/session/new")
async def new_session():
    return {"session_id": str(uuid.uuid4())}

@app.get("/reminders/{session_id}")
async def list_reminders(session_id: str):
    return {"reminders": get_reminders_for_session(session_id)}

@app.get("/search")
async def direct_search(q: str, n: int = 5):
    from tools.search import search_web
    return await search_web(q, num_results=n)

# ── voice endpoints ───────────────────────────────────────────

@app.post("/voice/transcribe")
async def voice_transcribe(
    audio: UploadFile = File(...)
):
    """Receive audio from browser mic, return transcript."""
    audio_bytes = await audio.read()
    extension   = audio.filename.split(".")[-1] if audio.filename else "webm"

    print(f"[VOICE] received {len(audio_bytes)} bytes "
          f"({audio.content_type})")

    result = transcribe(audio_bytes, extension=extension)

    if not result["success"] or not result["text"]:
        return {"success": False, "text": "", "error": "no speech detected"}

    return {
        "success":  True,
        "text":     result["text"],
        "language": result.get("language", "en"),
        "duration": result.get("duration", 0),
    }


@app.post("/voice/speak")
async def voice_speak(
    text:  str = Form(...),
    voice: str = Form(default="orion")
):
    """Convert text to speech, return WAV audio."""
    if not text.strip():
        return Response(status_code=400)

    audio_bytes = synthesize(text, voice_name=voice)

    if not audio_bytes:
        return Response(status_code=500)

    return Response(
        content      = audio_bytes,
        media_type   = "audio/wav",
        headers      = {
            "Content-Disposition": "inline; filename=speech.wav"
        }
    )


@app.post("/voice/chat")
async def voice_chat(
    session_id: str        = Form(...),
    audio:      UploadFile = File(...),
    voice:      str        = Form(default="orion"),
    speak_reply: str       = Form(default="true")
):
    """
    Full voice round-trip:
    audio in → transcribe → chat → speak reply → audio out
    """
    # 1. transcribe
    audio_bytes = await audio.read()
    extension   = "webm"
    stt_result  = transcribe(audio_bytes, extension=extension)

    if not stt_result["success"] or not stt_result["text"]:
        return {"success": False, "error": "no speech detected"}

    user_text = stt_result["text"]
    print(f"[VOICE CHAT] heard: '{user_text}'")

    # 2. run through normal chat pipeline
    normalized  = normalize(user_text)
    intent      = classify(normalized)
    result      = await route(intent, session_id)
    reply       = result["reply"]

    save_message(session_id, "user",      user_text)
    save_message(session_id, "assistant", reply)

    # 3. speak reply if requested
    audio_response = None
    if speak_reply.lower() == "true":
        audio_bytes_out = synthesize(reply, voice_name=voice)
        if audio_bytes_out:
            import base64
            audio_response = base64.b64encode(
                audio_bytes_out
            ).decode()

    return {
        "success":     True,
        "heard":       user_text,
        "reply":       reply,
        "audio_b64":   audio_response,
        "voice":       voice,
        "debug":       result.get("debug", {})
    }


@app.get("/voice/voices")
async def get_voices():
    """List available voices and their download status."""
    return {"voices": list_voices()}