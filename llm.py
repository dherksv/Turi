import httpx
import os
from dotenv import load_dotenv
from context import build_system_prompt

load_dotenv()
LLAMA_URL  = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-E2B-it-Q4_K_M")


async def chat(messages: list[dict], user_message: str) -> str:
    """
    messages     : conversation history (role/content dicts)
    user_message : the current user message — used to retrieve
                   relevant memories for context injection
    """
    system_prompt = build_system_prompt(user_message)

    full_messages = (
        [{"role": "system", "content": system_prompt}]
        + messages
    )

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{LLAMA_URL}/v1/chat/completions",
            json={
                "model":       MODEL_NAME,
                "messages":    full_messages,
                "temperature": 0.7,
                "max_tokens":  1024,
                "stream":      False,
            }
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


async def health_check() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{LLAMA_URL}/health")
            return r.status_code == 200
    except Exception:
        return False