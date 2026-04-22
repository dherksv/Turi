import sys
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
from fastapi.responses import StreamingResponse
from llm import chat, stream_chat, health_check
from mcp import init_mcp
from mcp import call as mcp_call, list_servers as mcp_list
from pipeline import init_audit_db, generate_report, audit_log

from debug_logger import (
    log_event, log_classification,
    get_recent_logs, get_error_logs
)
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

if sys.platform == "win32":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

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
    init_audit_db()
    init_mcp()          
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
        "tool_result": result.get("tool_result"),
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

# ── streaming text endpoint ───────────────────────────────────
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "empty message")

    import json as _json
    import time

    async def generate():
        task_start = time.time()
        task_id    = str(uuid.uuid4())

        try:
            normalized = normalize(req.message)
            intent     = classify(normalized)

            # log classification
            log_classification(
                req.message, intent, req.session_id
            )

            print(f"\n[STREAM] '{req.message}'")
            print(f"[STREAM] → {intent['intent_type']} | "
                  f"tool={intent['tool']} | "
                  f"conf={intent.get('confidence', 0):.2f}")

            # send classification to frontend for debug
            yield (f"data: {_json.dumps({'debug_intent': intent})}"
                   f"\n\n")

            tool_result = None

            if (intent["intent_type"] == "command"
                    and intent["tool"]
                    and intent["confidence"] >= 0.70):

                tool_start = time.time()

                log_event("tool_dispatch", "router", {
                    "tool":    intent["tool"],
                    "message": req.message[:100]
                }, session_id=req.session_id, task_id=task_id)

                from tools.registry import dispatch
                tool_result = await dispatch(
                    intent["tool"], intent
                )

                tool_ms = int((time.time() - tool_start) * 1000)

                from debug_logger import log_tool_call
                log_tool_call(
                    tool        = intent["tool"],
                    args        = intent,
                    result      = tool_result,
                    agent       = "executor",
                    session_id  = req.session_id,
                    task_id     = task_id,
                    duration_ms = tool_ms,
                    error       = tool_result.get("error")
                                  if tool_result else None
                )

                print(f"[TOOL] {intent['tool']} → "
                      f"status={tool_result.get('status')} "
                      f"({tool_ms}ms)")

                yield (f"data: {_json.dumps({'tool_result': tool_result, 'debug_tool_ms': tool_ms})}"
                       f"\n\n")

            history = get_history(req.session_id, limit=10)
            history.append({
                "role":    "user",
                "content": req.message
            })

            if tool_result and tool_result.get("status") == "ok":
                import json as j
                context = (
                    f"Tool '{intent['tool']}' result:\n"
                    f"{j.dumps(tool_result)[:2000]}\n\n"
                    f"Based on this, reply to: '{req.message}'"
                )
                history.append({
                    "role":    "system",
                    "content": context
                })

            full_reply = []
            llm_start  = time.time()

            async for chunk in stream_chat(
                messages     = history,
                user_message = req.message,
                input_mode   = "text"
            ):
                full_reply.append(chunk)
                yield (f"data: {_json.dumps({'chunk': chunk})}"
                       f"\n\n")

            llm_ms   = int((time.time() - llm_start) * 1000)
            complete = "".join(full_reply)

            save_message(req.session_id, "user",
                         req.message)
            save_message(req.session_id, "assistant",
                         complete)

            total_ms = int((time.time() - task_start) * 1000)

            log_event("request_complete", "orchestrator", {
                "message":   req.message[:100],
                "tool":      intent.get("tool"),
                "reply_len": len(complete),
                "llm_ms":    llm_ms,
                "total_ms":  total_ms
            }, session_id=req.session_id, task_id=task_id)

            yield (f"data: {_json.dumps({'done': True, 'full': complete, 'debug': {'total_ms': total_ms, 'llm_ms': llm_ms, 'tool': intent.get('tool'), 'tool_status': tool_result.get('status') if tool_result else None}})}"
                   f"\n\n")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log_event("stream_error", "router", {
                "message": req.message[:100]
            }, error=tb,
               session_id=req.session_id,
               task_id=task_id)
            print(f"[STREAM ERROR] {e}\n{tb}")
            yield (f"data: {_json.dumps({'error': str(e)})}"
                   f"\n\n")

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no"
        }
    )
# ── debug endpoints ───────────────────────────────────────────

@app.get("/debug/logs")
async def debug_logs(limit: int = 50):
    """Recent structured debug logs."""
    return {"logs": get_recent_logs(limit)}

@app.get("/debug/errors")
async def debug_errors(limit: int = 20):
    """Recent error logs."""
    return {"errors": get_error_logs(limit)}

@app.get("/debug/fs-roots")
async def debug_fs_roots():
    """Show which directories file search scans."""
    from mcp.filesystem_server import get_search_roots
    roots = get_search_roots()
    return {
        "roots": [str(r) for r in roots],
        "exist": [str(r) for r in roots if r.exists()]
    }

@app.post("/debug/classify")
async def debug_classify(req: ChatRequest):
    normalized = normalize(req.message)
    intent     = classify(normalized)
    log_classification(req.message, intent, req.session_id)
    return {
        "input":       req.message,
        "intent":      intent,
        "explanation": (
            f"→ {intent['intent_type']} | "
            f"tool={intent['tool']} | "
            f"conf={intent['confidence']}"
        )
    }

@app.get("/debug/test-youtube")
async def test_youtube(q: str = "funny cat"):
    """Test YouTube search directly."""
    from mcp import call as mcp_call
    result = await mcp_call("youtube", "search_videos", {
        "query":       q,
        "max_results": 3,
        "type":        "video"
    })
    return result

@app.get("/debug/test-amazon")
async def test_amazon(q: str = "headset", price: int = 5000):
    """Test Amazon search directly."""
    from mcp import call as mcp_call
    result = await mcp_call("amazon", "search_products", {
        "query":       q,
        "max_price":   price,
        "max_results": 3
    })
    return result

@app.get("/debug/test-files")
async def test_files(q: str = "report"):
    """Test file search directly."""
    from mcp import call as mcp_call
    result = await mcp_call("filesystem", "search_files", {
        "query": q
    })
    return result
# ── streaming voice endpoint ──────────────────────────────────

@app.post("/voice/stream")
async def voice_stream(
    session_id:  str        = Form(...),
    audio:       UploadFile = File(...),
    voice:       str        = Form(default="orion"),
):
    """
    Voice streaming pipeline:
    audio → STT → LLM stream → TTS sentence by sentence → audio chunks back

    Returns SSE with:
    - text chunks as LLM generates
    - audio chunks (base64 WAV) as each sentence completes
    """
    from voice.stt import transcribe
    from voice.tts import synthesize
    import json, base64, re

    # 1. transcribe audio
    audio_bytes = await audio.read()
    stt_result  = transcribe(audio_bytes, extension="webm")

    if not stt_result["success"] or not stt_result["text"]:
        async def err():
            yield f"data: {json.dumps({'error': 'no speech detected'})}\n\n"
        return StreamingResponse(err(), media_type="text/event-stream")

    user_text = stt_result["text"]
    print(f"[VOICE STREAM] heard: '{user_text}'")

    history = get_history(session_id, limit=10)
    history.append({"role": "user", "content": user_text})

    async def generate():
        import json, base64

        # send transcription immediately so UI can show it
        yield f"data: {json.dumps({'heard': user_text})}\n\n"

        full_reply    = []
        sentence_buf  = ""
        # sentence boundary — speak when we hit . ! ? or long pause
        sentence_end  = re.compile(r'[.!?]')

        async for chunk in stream_chat(
            messages     = history,
            user_message = user_text,
            input_mode   = "voice"
        ):
            full_reply.append(chunk)
            sentence_buf += chunk

            # send text chunk to UI immediately
            yield f"data: {json.dumps({'chunk': chunk})}\n\n"

            # check if we have a complete sentence
            if sentence_end.search(chunk) and len(sentence_buf.strip()) > 10:
                sentence = sentence_buf.strip()
                sentence_buf = ""

                # synthesize this sentence
                audio_bytes_out = synthesize(sentence, voice_name=voice)
                if audio_bytes_out:
                    audio_b64 = base64.b64encode(
                        audio_bytes_out
                    ).decode()
                    yield f"data: {json.dumps({'audio_chunk': audio_b64, 'sentence': sentence})}\n\n"

        # speak any remaining buffer
        if sentence_buf.strip():
            audio_bytes_out = synthesize(
                sentence_buf.strip(), voice_name=voice
            )
            if audio_bytes_out:
                audio_b64 = base64.b64encode(audio_bytes_out).decode()
                yield f"data: {json.dumps({'audio_chunk': audio_b64})}\n\n"

        complete_reply = "".join(full_reply)

        # save to memory
        save_message(session_id, "user",      user_text)
        save_message(session_id, "assistant", complete_reply)

        # done
        yield f"data: {json.dumps({'done': True, 'full': complete_reply})}\n\n"

    return StreamingResponse(
        generate(),
        media_type = "text/event-stream",
        headers    = {"Cache-Control": "no-cache",
                      "X-Accel-Buffering": "no"}
    )

# in main.py — already exists but let's make it more useful
@app.post("/debug/classify")
async def debug_classify(req: ChatRequest):
    from normalizer import normalize
    from intent import classify
    normalized = normalize(req.message)
    intent     = classify(normalized)
    return {
        "input":       req.message,
        "normalized":  normalized,
        "intent":      intent,
        "explanation": (
            f"Classified as '{intent['intent_type']}' "
            f"with tool='{intent['tool']}' "
            f"confidence={intent['confidence']}"
        )
    }
@app.get("/voice/voices")
async def get_voices():
    """List available voices and their download status."""
    return {"voices": list_voices()}

# ── audit endpoints ───────────────────────────────────────────

@app.get("/audit/report")
async def get_audit_report(session_id: str = None):
    """Generate and return an audit report."""
    report = await generate_report(session_id)
    return {"report": report}

@app.get("/audit/log")
async def get_audit_log(
    session_id: str = None,
    limit:      int = 50
):
    """Get raw audit log entries."""
    import sqlite3
    from pipeline.auditor import AUDIT_DB
    conn  = sqlite3.connect(AUDIT_DB)
    if session_id:
        rows = conn.execute("""
            SELECT timestamp, event_type, actor,
                   action, outcome, risk_level, concern
            FROM audit_log
            WHERE session_id=?
            ORDER BY id DESC LIMIT ?
        """, (session_id, limit)).fetchall()
    else:
        rows = conn.execute("""
            SELECT timestamp, event_type, actor,
                   action, outcome, risk_level, concern
            FROM audit_log
            ORDER BY id DESC LIMIT ?
        """, (limit,)).fetchall()
    conn.close()
    return {"logs": [
        {
            "timestamp":  r[0],
            "event_type": r[1],
            "actor":      r[2],
            "action":     r[3],
            "outcome":    r[4],
            "risk_level": r[5],
            "concern":    r[6]
        }
        for r in rows
    ]}

@app.get("/audit/concerns")
async def get_concerns(session_id: str = None):
    """Get all unresolved concerns."""
    import sqlite3
    from pipeline.auditor import AUDIT_DB
    conn = sqlite3.connect(AUDIT_DB)
    if session_id:
        rows = conn.execute("""
            SELECT id, timestamp, concern_type,
                   description, severity
            FROM concerns
            WHERE session_id=? AND resolved=0
            ORDER BY id DESC
        """, (session_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, timestamp, concern_type,
                   description, severity
            FROM concerns
            WHERE resolved=0 ORDER BY id DESC LIMIT 50
        """).fetchall()
    conn.close()
    return {"concerns": [
        {
            "id":           r[0],
            "timestamp":    r[1],
            "concern_type": r[2],
            "description":  r[3],
            "severity":     r[4]
        }
        for r in rows
    ]}

@app.get("/agents/status")
async def agents_status():
    """Check all agents health."""
    from agents import (
        chat_agent, fast_agent,
        validator_agent, monitor_agent, auditor_agent
    )
    return {
        "orchestrator": {
            "name":   "Gemma 4 E2B",
            "port":   8081,
            "status": "ok" if await chat_agent.is_online() else "offline"
        },
        "memory_guard": {
            "name":   "Qwen 2.5 1.5B",
            "port":   8082,
            "status": "ok" if await fast_agent.is_online() else "offline"
        },
        "validator": {
            "name":   "Phi-4 Mini",
            "port":   8083,
            "status": "ok" if await validator_agent.is_online() else "offline"
        },
        "monitor": {
            "name":   "Llama 3.2 1B",
            "port":   8084,
            "status": "ok" if await monitor_agent.is_online() else "offline"
        },
        "auditor": {
            "name":   "Qwen 2.5 1.5B",
            "port":   8082,
            "status": "ok" if await auditor_agent.is_online() else "offline"
        }
    }
# ── Memory management endpoints ──────────────────────────────────

@app.get("/memory/status")
async def memory_status(session_id: str = ""):
    """Show memory layer activity — who wrote what."""
    conn = sqlite3.connect("data/assistant.db")

    scratchpad_recent = conn.execute("""
        SELECT agent_name, result_type, written_at
        FROM task_scratchpad
        ORDER BY id DESC LIMIT 20
    """).fetchall()

    conn.close()

    # gateway write log from pipeline
    from pipeline.agent_memory import task_memory_factory

    return {
        "scratchpad_recent": [
            {
                "agent":       r[0],
                "result_type": r[1],
                "written_at":  r[2]
            }
            for r in scratchpad_recent
        ],
        "memory_layers": {
            "tier_1_working":   "private per agent — not visible",
            "tier_2_scratchpad":"SQLite task_scratchpad table",
            "tier_3_long_term": "ChromaDB + user_profile.json"
        },
        "write_rule": (
            "ALL writes to tier 2/3 go through "
            "Memory Guard (Qwen 1.5B) first"
        )
    }

# ── MCP endpoints ──────────────────────────────────

@app.get("/mcp/servers")
async def get_mcp_servers():
    """List all registered MCP servers and their tools."""
    return {"servers": mcp_list()}

@app.post("/mcp/call")
async def call_mcp_tool(req: dict):
    """
    Call any MCP tool directly.
    Body: { server, tool, args }
    """
    server = req.get("server")
    tool   = req.get("tool")
    args   = req.get("args", {})

    if not server or not tool:
        raise HTTPException(400, "server and tool required")

    result = await mcp_call(server, tool, args)
    return result