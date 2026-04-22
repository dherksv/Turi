import os
from dotenv import load_dotenv
from agents.base import BaseAgent, AgentConfig

load_dotenv()

validator_agent = BaseAgent(AgentConfig(
    name        = "validator",
    model       = os.getenv("VALIDATOR_MODEL",
                            "Phi-4-mini-instruct-Q4_K_M"),
    server_url  = os.getenv("VALIDATOR_URL",
                            "http://localhost:8083"),
    temperature = 0.1,   # very low — we want consistent judgments
    max_tokens  = 256,
    timeout     = 15,
    system      = (
        "You are a strict AI safety validator. "
        "Your only job is to check if an AI intent is safe "
        "and matches what the user asked. "
        "Always respond with valid JSON only. "
        "Never add explanation outside the JSON."
    )
))