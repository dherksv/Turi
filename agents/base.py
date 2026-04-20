import httpx
import json
from dataclasses import dataclass


@dataclass
class AgentConfig:
    name:        str
    model:       str
    server_url:  str
    temperature: float = 0.7
    max_tokens:  int   = 1024
    timeout:     int   = 60
    system:      str   = ""


class BaseAgent:
    """
    A single LLM agent pointing at a specific
    model on a specific llama-server instance.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    async def chat(
        self,
        messages:    list[dict],
        system:      str  = "",
        temperature: float | None = None,
        max_tokens:  int  | None = None,
    ) -> str:
        system_prompt = system or self.config.system
        full_messages = []
        if system_prompt:
            full_messages.append(
                {"role": "system", "content": system_prompt}
            )
        full_messages.extend(messages)

        try:
            async with httpx.AsyncClient(
                timeout=self.config.timeout
            ) as client:
                resp = await client.post(
                    f"{self.config.server_url}/v1/chat/completions",
                    json={
                        "model":       self.config.model,
                        "messages":    full_messages,
                        "temperature": temperature or self.config.temperature,
                        "max_tokens":  max_tokens  or self.config.max_tokens,
                        "stream":      False,
                    }
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]

        except httpx.TimeoutException:
            print(f"[{self.config.name}] timeout")
            return f"[{self.config.name} timed out]"
        except httpx.ConnectError:
            print(f"[{self.config.name}] cannot connect to "
                  f"{self.config.server_url}")
            return f"[{self.config.name} unreachable]"
        except Exception as e:
            print(f"[{self.config.name}] error: {e}")
            return f"[{self.config.name} error: {e}]"

    async def is_online(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(
                    f"{self.config.server_url}/health"
                )
                return r.status_code == 200
        except Exception:
            return False