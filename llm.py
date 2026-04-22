import httpx
import json
import os
from dotenv import load_dotenv
from context import build_system_prompt
from agents  import chat_agent, fast_agent

load_dotenv()


async def chat(
    messages:     list[dict],
    user_message: str,
    input_mode:   str = "text"
) -> str:
    """Standard single-turn chat — waits for full reply."""
    system = build_system_prompt(user_message, input_mode=input_mode)
    return await chat_agent.chat(messages, system=system)


async def stream_chat(
    messages:     list[dict],
    user_message: str,
    input_mode:   str = "text"
):
    """
    Async generator — yields text chunks as LLM generates.
    Usage:
        async for chunk in stream_chat(messages, user_message):
            print(chunk, end="", flush=True)
    """
    system        = build_system_prompt(user_message, input_mode=input_mode)
    full_messages = [{"role": "system", "content": system}] + messages

    server_url = os.getenv("LLAMA_SERVER_URL", "http://localhost:8081")
    model      = os.getenv("MODEL_NAME",       "gemma-4-E4B-it-Q4_K_M")

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST",
                f"{server_url}/v1/chat/completions",
                json={
                    "model":       model,
                    "messages":    full_messages,
                    "temperature": 0.7,
                    "max_tokens":  1024,
                    "stream":      True,
                }
            ) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # strip "data: "

                    if data.strip() == "[DONE]":
                        break

                    try:
                        obj   = json.loads(data)
                        delta = obj["choices"][0]["delta"]
                        text  = delta.get("content", "")
                        if text:
                            yield text
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

    except httpx.ConnectError:
        print(f"[LLM] stream_chat — cannot connect to {server_url}")
        yield "[connection error — is llama-server running?]"
    except httpx.TimeoutException:
        print(f"[LLM] stream_chat — timeout")
        yield "[response timed out]"
    except Exception as e:
        print(f"[LLM] stream_chat — error: {e}")
        yield f"[error: {e}]"


async def fast_chat(prompt: str) -> str:
    """
    Quick single-turn call using the smaller fast agent.
    Falls back to chat_agent if fast_agent is offline.
    """
    online = await fast_agent.is_online()
    if not online:
        print("[LLM] fast_agent offline — falling back to chat_agent")
        return await chat_agent.chat(
            [{"role": "user", "content": prompt}]
        )
    return await fast_agent.chat(
        [{"role": "user", "content": prompt}]
    )


async def health_check() -> dict:
    return {
        "chat_agent": "ok" if await chat_agent.is_online() else "offline",
        "fast_agent": "ok" if await fast_agent.is_online() else "offline",
    }