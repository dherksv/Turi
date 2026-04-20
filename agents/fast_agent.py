import os
from dotenv import load_dotenv
from agents.base import BaseAgent, AgentConfig

load_dotenv()

fast_agent = BaseAgent(AgentConfig(
    name        = "fast_agent",
    model       = os.getenv("FAST_MODEL", "qwen2.5-1.5b"),
    server_url  = os.getenv("FAST_SERVER_URL", "http://localhost:8082"),
    temperature = 0.3,   # lower — we want deterministic outputs
    max_tokens  = 256,   # short outputs only
    timeout     = 20,
    system      = (
        "You are a precise AI classifier and validator. "
        "Give short, structured answers only. "
        "Never add unnecessary explanation."
    )
))