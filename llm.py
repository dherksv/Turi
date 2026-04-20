import httpx
import os
from dotenv import load_dotenv
from context import build_system_prompt
from agents  import chat_agent, fast_agent

load_dotenv()
LLAMA_URL  = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8081")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma-4-E2B-it-Q4_K_M")

async def chat(messages: list[dict], user_message: str) -> str:
    """Main conversation — uses gemma4 chat agent."""
    system = build_system_prompt(user_message)
    return await chat_agent.chat(messages, system=system)
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


async def fast_chat(prompt: str) -> str:
    """
    Quick single-turn call — uses qwen fast agent.
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