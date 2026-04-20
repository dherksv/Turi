import os
from dotenv import load_dotenv
from agents.base import BaseAgent, AgentConfig

load_dotenv()

chat_agent = BaseAgent(AgentConfig(
    name       = "chat_agent",
    model      = os.getenv("MODEL_NAME", "gemma-4-E2B-it-Q4_K_M"),
    server_url = os.getenv("LLAMA_SERVER_URL", "http://localhost:8081"),
    temperature = 0.7,
    max_tokens  = 1024,
    timeout     = 120,
    system      = (
        "You are a helpful personal assistant. "
        "You are concise, honest, and direct. "
        "Never make up facts. If unsure, say so."
    )
))