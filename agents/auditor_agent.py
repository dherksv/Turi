import os
from dotenv import load_dotenv
from agents.base import BaseAgent, AgentConfig

load_dotenv()

auditor_agent = BaseAgent(AgentConfig(
    name        = "auditor",
    model       = os.getenv("AUDITOR_MODEL",
                            "qwen2.5-1.5b-instruct-q4_k_m"),
    server_url  = os.getenv("AUDITOR_URL",
                            "http://localhost:8082"),
    temperature = 0.1,
    max_tokens  = 512,
    timeout     = 20,
    system      = (
        "You are an AI audit system. "
        "Review AI actions and flag concerns. "
        "Respond with JSON only. No extra text."
    )
))