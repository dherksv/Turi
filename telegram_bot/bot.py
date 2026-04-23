"""
Turi Telegram Bot
Receives messages and voice from Telegram,
runs through full Turi pipeline, replies back.
"""

import os
import json
import asyncio
import tempfile
import subprocess
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import httpx

from fastapi import FastAPI, Request, Header
from normalizer import normalize
from intent     import classify
from router     import route
from memory     import (
    init_db, save_message,
    get_history, save_reminder
)
from voice.stt  import transcribe
from voice.tts  import synthesize
from debug_logger import log_event, log_error

load_dotenv()

TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
API_URL = f"https://api.telegram.org/bot{TOKEN}"
USE_VOICE = os.getenv("TURI_TELEGRAM_VOICE", "true").lower() == "true"

# map telegram chat_id → Turi session_id
# persisted to file so it survives restarts
SESSION_MAP_PATH = Path("data/telegram_sessions.json")


def load_session_map() -> dict:
    if SESSION_MAP_PATH.exists():
        try:
            return json.loads(SESSION_MAP_PATH.read_text())
        except Exception:
            return {}
    return {}


def save_session_map(mapping: dict):
    SESSION_MAP_PATH.write_text(
        json.dumps(mapping, indent=2)
    )


def get_session_id(chat_id: int) -> str:
    """Get or create a Turi session for this Telegram chat."""
    import uuid
    mapping  = load_session_map()
    key      = str(chat_id)
    if key not in mapping:
        mapping[key] = str(uuid.uuid4())
        save_session_map(mapping)
        log_event("telegram_new_session", "telegram_bot", {
            "chat_id":    chat_id,
            "session_id": mapping[key]
        })
    return mapping[key]


# ── Telegram API calls ────────────────────────────────────────

async def send_message(
    chat_id: int,
    text:    str,
    parse_mode: str = "Markdown"
) -> dict:
    """Send a text message to Telegram."""
    # Telegram has 4096 char limit
    if len(text) > 4000:
        text = text[:3990] + "\n...[truncated]"

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{API_URL}/sendMessage",
            json={
                "chat_id":    chat_id,
                "text":       text,
                "parse_mode": parse_mode
            }
        )
        return resp.json()


async def send_voice(
    chat_id:     int,
    ogg_bytes:   bytes,
    caption:     str = ""
) -> dict:
    """Send a voice message (OGG Opus) to Telegram."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{API_URL}/sendVoice",
            data    = {
                "chat_id": str(chat_id),
                "caption": caption[:1000] if caption else ""
            },
            files   = {
                "voice": ("voice.ogg", ogg_bytes,
                          "audio/ogg")
            }
        )
        return resp.json()


async def send_typing(chat_id: int):
    """Show 'typing...' indicator."""
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            f"{API_URL}/sendChatAction",
            json={
                "chat_id": chat_id,
                "action":  "typing"
            }
        )


async def send_voice_recording(chat_id: int):
    """Show 'recording voice...' indicator."""
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            f"{API_URL}/sendChatAction",
            json={
                "chat_id": chat_id,
                "action":  "record_voice"
            }
        )


async def download_file(file_id: str) -> bytes | None:
    """Download a file from Telegram servers."""
    async with httpx.AsyncClient(timeout=30) as client:
        # get file path
        resp = await client.get(
            f"{API_URL}/getFile",
            params={"file_id": file_id}
        )
        data = resp.json()
        if not data.get("ok"):
            return None

        file_path = data["result"]["file_path"]
        file_url  = (f"https://api.telegram.org/"
                     f"file/bot{TOKEN}/{file_path}")

        # download the file
        download = await client.get(file_url)
        return download.content


# ── Audio conversion ──────────────────────────────────────────

def wav_to_ogg(wav_bytes: bytes) -> bytes | None:
    """
    Convert WAV (Piper output) to OGG Opus
    (required by Telegram voice messages).
    Requires ffmpeg in PATH.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".wav", delete=False
    ) as wav_file:
        wav_file.write(wav_bytes)
        wav_path = wav_file.name

    ogg_path = wav_path.replace(".wav", ".ogg")

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i",        wav_path,
            "-c:a",      "libopus",
            "-b:a",      "32k",
            "-vbr",      "on",
            "-compression_level", "10",
            "-ar",       "48000",
            "-ac",       "1",
            ogg_path
        ], capture_output=True, timeout=30)

        if result.returncode != 0:
            print(f"[TELEGRAM] ffmpeg error: "
                  f"{result.stderr.decode()[:200]}")
            return None

        return Path(ogg_path).read_bytes()

    except FileNotFoundError:
        print("[TELEGRAM] ffmpeg not found — "
              "install ffmpeg and add to PATH")
        return None
    except Exception as e:
        print(f"[TELEGRAM] wav→ogg error: {e}")
        return None
    finally:
        for p in [wav_path, ogg_path]:
            try:
                Path(p).unlink()
            except Exception:
                pass


def ogg_to_wav(ogg_bytes: bytes) -> bytes | None:
    """
    Convert Telegram voice note (OGG Opus)
    to WAV for Whisper transcription.
    """
    with tempfile.NamedTemporaryFile(
        suffix=".ogg", delete=False
    ) as f:
        f.write(ogg_bytes)
        ogg_path = f.name

    wav_path = ogg_path.replace(".ogg", ".wav")

    try:
        result = subprocess.run([
            "ffmpeg", "-y",
            "-i",   ogg_path,
            "-ar",  "16000",
            "-ac",  "1",
            "-f",   "wav",
            wav_path
        ], capture_output=True, timeout=30)

        if result.returncode != 0:
            return None

        return Path(wav_path).read_bytes()

    except Exception as e:
        print(f"[TELEGRAM] ogg→wav error: {e}")
        return None
    finally:
        for p in [ogg_path, wav_path]:
            try:
                Path(p).unlink()
            except Exception:
                pass


# ── Message processing ────────────────────────────────────────

async def process_text_message(
    chat_id:  int,
    text:     str,
    username: str = ""
) -> str:
    """Run text through full Turi pipeline."""

    # deferred imports — avoids circular import issues
    from normalizer import normalize
    from intent     import classify
    from router     import route
    from memory     import save_message

    session_id = get_session_id(chat_id)

    print(f"[TELEGRAM] processing: '{text[:80]}'")

    try:
        normalized = normalize(text)
        intent     = classify(normalized)

        print(f"[TELEGRAM] classified → "
              f"{intent['intent_type']} | "
              f"tool={intent['tool']} | "
              f"conf={intent.get('confidence', 0):.2f}")

        result = await route(
            intent     = intent,
            session_id = session_id,
            input_mode = "text"
        )

        reply = result.get(
            "reply",
            "I'm not sure how to help with that."
        )

        save_message(session_id, "user",      text)
        save_message(session_id, "assistant", reply)

        print(f"[TELEGRAM] reply: '{reply[:80]}'")
        return reply

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[TELEGRAM] pipeline error: {e}")
        print(f"[TELEGRAM] traceback:\n{tb}")
        raise  # re-raise so handle_update catches it


async def process_voice_message(
    chat_id:  int,
    file_id:  str,
    username: str = ""
) -> tuple[str, str]:
    """Process voice note from Telegram."""

    from voice.stt import transcribe
    from voice.tts import synthesize

    session_id = get_session_id(chat_id)

    print(f"[TELEGRAM] processing voice from {chat_id}")

    # download OGG from Telegram
    ogg_bytes = await download_file(file_id)
    if not ogg_bytes:
        return "", "Sorry, I couldn't download your voice message."

    # convert OGG → WAV for Whisper
    wav_bytes = ogg_to_wav(ogg_bytes)
    if not wav_bytes:
        return "", ("Voice conversion failed. "
                    "Make sure ffmpeg is installed.")

    # transcribe
    stt_result = transcribe(wav_bytes, extension="wav")
    if not stt_result.get("success") or not stt_result.get("text"):
        return "", "I couldn't understand that voice message."

    transcript = stt_result["text"]
    print(f"[TELEGRAM] transcribed: '{transcript}'")

    reply = await process_text_message(
        chat_id, transcript, username
    )
    return transcript, reply



# ── Main webhook handler ──────────────────────────────────────

async def handle_update(update: dict):
    """
    Process one Telegram update (message or voice).
    Called by the FastAPI webhook endpoint.
    """
    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat_id  = message["chat"]["id"]
    username = message.get("from", {}).get("first_name", "")
    msg_type = "unknown"

    try:
        # ── voice message ─────────────────────────────────────
        if message.get("voice"):
            msg_type = "voice"
            await send_voice_recording(chat_id)

            file_id    = message["voice"]["file_id"]
            transcript, reply = await process_voice_message(
                chat_id, file_id, username
            )

            if not reply:
                await send_message(chat_id,
                    "I couldn't process that voice message.")
                return

            # send text reply first (fast)
            transcript_note = (
                f"_You said: {transcript}_\n\n"
                if transcript else ""
            )
            await send_message(
                chat_id,
                transcript_note + reply
            )

            # send voice reply if enabled
            if USE_VOICE and reply:
                await send_voice_recording(chat_id)
                voice_name  = "orion"   # default voice
                wav_bytes   = synthesize(reply, voice_name)
                if wav_bytes:
                    ogg_bytes = wav_to_ogg(wav_bytes)
                    if ogg_bytes:
                        await send_voice(chat_id, ogg_bytes)

        # ── text message ──────────────────────────────────────
        elif message.get("text"):
            msg_type = "text"
            text     = message["text"]

            # handle bot commands
            if text.startswith("/"):
                await handle_command(chat_id, text, username)
                return

            print(f"[TELEGRAM] text from {username} "
                  f"({chat_id}): '{text[:60]}'")

            # show typing
            await send_typing(chat_id)

            # process through pipeline
            reply = await process_text_message(
                chat_id, text, username
            )

            # send reply
            await send_message(chat_id, reply)

            # voice reply if enabled and reply is short
            if USE_VOICE and reply and len(reply) < 600:
                try:
                    from voice.tts import synthesize
                    await send_voice_recording(chat_id)
                    wav_bytes = synthesize(reply, "orion")
                    if wav_bytes:
                        ogg_bytes = wav_to_ogg(wav_bytes)
                        if ogg_bytes:
                            await send_voice(
                                chat_id, ogg_bytes
                            )
                except Exception as ve:
                    print(f"[TELEGRAM] voice reply "
                          f"error: {ve}")
                    # voice failed but text was sent — ok

        # ── photo / document (future) ─────────────────────────
        elif message.get("document"):
            await send_message(
                chat_id,
                "I received your file. "
                "File handling via Telegram coming soon!"
            )

        log_event("telegram_handled", "telegram_bot", {
            "chat_id":  chat_id,
            "msg_type": msg_type,
            "username": username
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[TELEGRAM] ERROR for chat {chat_id}: {e}")
        print(f"[TELEGRAM] TRACEBACK:\n{tb}")
        log_error("telegram_bot", e, {
            "chat_id":  chat_id,
            "msg_type": msg_type
        })
        try:
            await send_message(
                chat_id,
                f"Something went wrong: `{type(e).__name__}: {str(e)[:200]}`"
            )
        except Exception:
            pass


async def handle_command(
    chat_id:  int,
    command:  str,
    username: str
):
    """Handle Telegram bot commands like /start /help."""
    cmd = command.split()[0].lower()

    if cmd == "/start":
        session_id = get_session_id(chat_id)
        await send_message(chat_id, f"""
*Hey {username}! I'm Turi* 🤖

Named after Alan Turing — father of computer science.

*What I can do:*
- Answer questions
- Search the web
- Find products on Amazon
- Play YouTube videos
- Set reminders
- Open files on your computer
- Speak replies as voice messages

*Commands:*
/start — this message
/help — show commands
/voice on|off — toggle voice replies
/clear — start fresh conversation
/turing — Alan Turing facts
/status — system status

Just send me a message or voice note to get started!
        """.strip())

    elif cmd == "/help":
        await send_message(chat_id, """
*Turi Commands:*

/start — introduction
/voice on — enable voice replies
/voice off — disable text-only mode
/clear — clear conversation history
/turing — Alan Turing tribute
/status — check which agents are online
/stop — stop current task

*Tips:*
- Send a voice note — I'll transcribe and reply
- Ask me to search, buy, play, or open anything
- I remember our conversation history
        """.strip())

    elif cmd == "/clear":
        # start a fresh session
        mapping  = load_session_map()
        import uuid
        mapping[str(chat_id)] = str(uuid.uuid4())
        save_session_map(mapping)
        await send_message(
            chat_id,
            "Fresh start! Conversation cleared. 🔄"
        )

    elif cmd == "/voice":
        parts = command.split()
        if len(parts) > 1:
            state = parts[1].lower()
            # store preference
            prefs_path = Path("data/telegram_prefs.json")
            prefs      = {}
            if prefs_path.exists():
                try:
                    prefs = json.loads(
                        prefs_path.read_text()
                    )
                except Exception:
                    pass
            prefs[str(chat_id)] = {"voice": state == "on"}
            prefs_path.write_text(json.dumps(prefs))
            await send_message(
                chat_id,
                f"Voice replies {'enabled 🔊' if state == 'on' else 'disabled 🔇'}"
            )

    elif cmd == "/turing":
        await send_message(chat_id, """
*Alan Mathison Turing* (1912–1954)

The man your assistant is named after.

- Invented the Turing Machine — the theoretical \
basis of every computer ever built
- Broke the Nazi Enigma cipher at Bletchley Park, \
credited with shortening WWII by 2 years
- Proposed the Turing Test in 1950 — still the \
defining measure of machine intelligence
- Nearly qualified for the 1948 British Olympic \
marathon team
- Prosecuted for his sexuality in 1952. \
Received a posthumous royal pardon in 2013.

_"We can only see a short distance ahead, but we \
can see plenty there that needs to be done."_

Every line of code in this system stands on \
the foundation he built.
        """.strip())

    elif cmd == "/status":
        from agents import (
            chat_agent, fast_agent,
            validator_agent, monitor_agent
        )
        chat_ok  = await chat_agent.is_online()
        fast_ok  = await fast_agent.is_online()
        valid_ok = await validator_agent.is_online()
        mon_ok   = await monitor_agent.is_online()

        status_text = f"""
*Turi System Status*

{'✅' if chat_ok  else '❌'} Orchestrator (Gemma 4)
{'✅' if valid_ok else '❌'} Validator (Phi-4 Mini)
{'✅' if fast_ok  else '❌'} Memory Guard (Qwen 1.5B)
{'✅' if mon_ok   else '❌'} Monitor (Llama 3.2 1B)
✅ Telegram connected
        """.strip()
        await send_message(chat_id, status_text)

    elif cmd == "/stop":
        session_id = get_session_id(chat_id)
        await send_message(
            chat_id, "Stopping current task... ⏹"
        )

    else:
        await send_message(
            chat_id,
            f"Unknown command: {cmd}\nTry /help"
        )


# ── Webhook setup ─────────────────────────────────────────────

async def set_webhook(webhook_url: str) -> dict:
    """Register webhook URL with Telegram."""
    secret = os.getenv("TELEGRAM_WEBHOOK_SECRET", "secret")
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            f"{API_URL}/setWebhook",
            json={
                "url":           webhook_url,
                "secret_token":  secret,
                "allowed_updates": ["message", "edited_message"]
            }
        )
        return resp.json()


async def delete_webhook() -> dict:
    """Remove webhook — switch to polling mode."""
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            f"{API_URL}/deleteWebhook"
        )
        return resp.json()


async def get_updates(offset: int = 0) -> list[dict]:
    """Long polling fallback when no webhook."""
    async with httpx.AsyncClient(timeout=35) as client:
        resp = await client.get(
            f"{API_URL}/getUpdates",
            params={
                "offset":  offset,
                "timeout": 30,
                "allowed_updates": ["message"]
            }
        )
        data = resp.json()
        return data.get("result", [])


async def polling_loop():
    """
    Long-polling fallback — use when running locally
    without a public URL for webhooks.
    """
    print("[TELEGRAM] starting polling loop...")
    offset = 0
    while True:
        try:
            updates = await get_updates(offset)
            for update in updates:
                await handle_update(update)
                offset = update["update_id"] + 1
        except Exception as e:
            print(f"[TELEGRAM] polling error: {e}")
            await asyncio.sleep(5)