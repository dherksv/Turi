import os
from dotenv import load_dotenv
from agents.base import BaseAgent, AgentConfig

load_dotenv()

monitor_agent = BaseAgent(AgentConfig(
    name        = "monitor",
    model       = os.getenv("MONITOR_MODEL",
                            "Llama-3.2-1B-Instruct-Q4_K_M"),
    server_url  = os.getenv("MONITOR_URL",
                            "http://localhost:8084"),
    temperature = 0.1,
    max_tokens  = 256,
    timeout     = 15,
    system      = (
        "You are a failure detection system. "
        "Analyze errors and classify them. "
        "Respond with JSON only. No extra text."
    )
))