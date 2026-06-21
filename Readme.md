```markdown
# Turi — Intelligent Multi-Agent Personal Assistant

> Named after **Alan Mathison Turing** (1912–1954) — father of computer science, codebreaker, and the man who asked the question this project tries to answer.

---

![Turi Screenshot](docs/screenshot.png)

---

## What is Turi?

Turi is a fully local, privacy-first multi-agent AI personal assistant that runs entirely on your own hardware — no cloud, no paid APIs, no data leaving your device.

It orchestrates multiple specialized language models across a structured pipeline that handles intent classification, validation, tool execution, memory protection, failure recovery, and audit logging — all from scratch, without orchestration frameworks.

---

## Features

- **Multi-agent architecture** — five specialized models (Gemma 4, Phi-4 Mini, Qwen 2.5, Llama 3.2) each handling a specific role
- **Voice interaction** — custom wake word detection ("Hey Turi") + Whisper STT + Piper TTS with Orion (male) and Lyra (female) voices
- **Web search** — self-hosted SearXNG, no Google API key needed
- **Amazon shopping** — real product search with price filters and Wilson score ranking
- **YouTube** — search and auto-open videos and music via yt-dlp
- **File system** — search, open, and read files; open Windows apps by name
- **Reminders** — natural language scheduling with browser popup notifications
- **Telegram integration** — full pipeline accessible via Telegram bot (text + voice)
- **Three-tier memory** — isolated working memory per agent, guarded shared writes via Qwen memory guard
- **Streaming responses** — word-by-word streaming with dual fast/deep path routing
- **Audit log** — append-only structured log of every agent decision
- **Failure recovery** — checkpoint-based task resumption, classified error handling

---

## Architecture

```
User input (text / voice / Telegram)
        ↓
Input normalizer → Intent classifier
        ↓
Validator — Phi-4 Mini (port 8083)
        ↓
Memory gateway — Qwen 1.5B guard (port 8082)
        ↓
MCP tool servers (Amazon · YouTube · Files · SearXNG)
        ↓
Failure monitor — Llama 3.2 1B (port 8084)
        ↓
Auditor — Qwen 1.5B (port 8082)
        ↓
Orchestrator — Gemma 4 E2B (port 8081)
        ↓
Output (Browser SSE · Piper TTS · Telegram · Notifications)
```

---

## Agent Roles

| Agent | Model | Role |
|---|---|---|
| Orchestrator | Gemma 4 E2B | Reasoning, planning, response generation |
| Validator | Phi-4 Mini | Intent safety and correctness check |
| Memory Guard | Qwen 2.5 1.5B | Shared memory write protection |
| Auditor | Qwen 2.5 1.5B | Post-execution concern flagging |
| Failure Monitor | Llama 3.2 1B | Error classification and recovery |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.13, FastAPI, uvicorn |
| LLM inference | llama-server (llama.cpp), GGUF quantized models |
| Vector memory | ChromaDB + sentence-transformers |
| Conversation memory | SQLite |
| Voice STT | faster-whisper (Whisper base) |
| Voice TTS | Piper TTS |
| Wake word | Custom CNN-GRU ONNX model |
| Web search | SearXNG (self-hosted Docker) |
| Browser automation | Playwright |
| Media search | yt-dlp |
| Telegram | Telegram Bot API via httpx |
| Frontend | Vanilla HTML/CSS/JS, SSE streaming |
| Audit | Append-only SQLite + JSONL logs |

---

## Hardware Requirements

| RAM | Recommended setup |
|---|---|
| 8 GB | Gemma 4 E2B Q4 + Qwen 1.5B shared — workable |
| 16 GB | Full stack comfortable, all agents simultaneous |
| 32 GB+ | Upgrade to larger orchestrator model |

Tested on: Intel i5 12th generation, 8GB RAM, Windows 11, CPU-only inference.

---

## Getting Started

### Prerequisites

```bash
# Python 3.11+
python --version

# Install dependencies
pip install fastapi uvicorn httpx python-multipart python-dotenv \
            chromadb sentence-transformers pytz pydantic \
            playwright lingua-language-detector networkx \
            apscheduler psutil faster-whisper onnxruntime \
            librosa python-telegram-bot openwakeword pyaudio \
            yt-dlp

playwright install chromium
```

### Models

Download GGUF models from Hugging Face and place in `models/`:

```bash
# Orchestrator
huggingface-cli download unsloth/gemma-4-E2B-it-GGUF \
  --include "gemma-4-E2B-it-Q4_K_M.gguf" --local-dir ./models/

# Validator
huggingface-cli download microsoft/Phi-4-mini-instruct-GGUF \
  --include "Phi-4-mini-instruct-Q4_K_M.gguf" --local-dir ./models/

# Memory guard + auditor
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
  --include "qwen2.5-1.5b-instruct-q4_k_m.gguf" --local-dir ./models/

# Failure monitor
huggingface-cli download bartowski/Llama-3.2-1B-Instruct-GGUF \
  --include "Llama-3.2-1B-Instruct-Q4_K_M.gguf" --local-dir ./models/
```

### Start model servers

```bash
# Terminal 1 — Orchestrator
llama-server.exe -m ./models/gemma-4-E2B-it-Q4_K_M.gguf --port 8081 --ctx-size 8192

# Terminal 2 — Memory guard + Auditor
llama-server.exe -m ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8082 --ctx-size 2048

# Terminal 3 — Validator
llama-server.exe -m ./models/Phi-4-mini-instruct-Q4_K_M.gguf --port 8083 --ctx-size 2048

# Terminal 4 — Failure monitor
llama-server.exe -m ./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf --port 8084 --ctx-size 2048
```

### Start SearXNG

```bash
docker run -d --name searxng -p 8888:8080 searxng/searxng
```

### Configure environment

Copy `.env.example` to `.env` and fill in your values:

```bash
LLAMA_SERVER_URL=http://localhost:8081
MODEL_NAME=gemma-4-E2B-it-Q4_K_M
FAST_SERVER_URL=http://localhost:8082
FAST_MODEL=qwen2.5-1.5b-instruct-q4_k_m
VALIDATOR_URL=http://localhost:8083
VALIDATOR_MODEL=Phi-4-mini-instruct-Q4_K_M
MONITOR_URL=http://localhost:8084
MONITOR_MODEL=Llama-3.2-1B-Instruct-Q4_K_M
SEARXNG_URL=http://localhost:8888
TELEGRAM_BOT_TOKEN=your_token_here
USER_TIMEZONE=Asia/Kolkata
USER_NAME=Your Name
```

Edit `data/user_profile.json` with your details.

### Run

```bash
uvicorn main:app --reload --port 3000
```

Open `http://localhost:3000` in your browser.

---

## Project Structure

```
assistant/
├── main.py                  # FastAPI application
├── normalizer.py            # Input cleaning
├── intent.py                # Rule-based classifier
├── router.py                # Pipeline orchestration
├── context.py               # Dynamic system prompt builder
├── llm.py                   # LLM client wrappers
├── llm_router.py            # Fast/deep path routing
├── agents/                  # Agent model configurations
├── pipeline/                # Validator, memory guard, monitor, auditor
├── mcp/                     # MCP tool servers
├── voice/                   # STT and TTS wrappers
├── wake_word/               # Wake word detector and model
├── telegram_bot/            # Telegram channel integration
├── memory.py                # SQLite conversation store
├── vector_memory.py         # ChromaDB semantic memory
├── sse.py                   # Server-Sent Events manager
├── scheduler.py             # Background reminder loop
├── debug_logger.py          # Structured JSONL logging
├── data/                    # Databases and user profile
├── logs/                    # Debug and audit logs
├── evaluation/              # Academic evaluation scripts
└── frontend/                # Single-page chat interface
```

---

## Evaluation Results

| Metric | Result |
|---|---|
| Intent classification accuracy | 90.6% |
| Tool routing accuracy | 87.5% |
| Macro F1-score | 0.913 |
| Human evaluation score | 4.10 / 5.00 |
| Fast path avg latency | 30.5s (CPU-only) |
| Wake word detection confidence | 0.92+ |

*Evaluated on Intel i5 12th gen, 8GB RAM, CPU inference only.*
*Best-case scenario conditions — all agents online, no resource contention.*

---

## Telegram Bot Commands

```
/start    — introduction and capabilities
/help     — list all commands
/voice on — enable voice replies
/voice off — text-only mode
/clear    — start fresh conversation
/turing   — Alan Turing tribute
/status   — check agent health
/stop     — cancel current task
```

---

## Example Interactions

```
You:  Hey Turi, find wireless headset under 5000 rupees
Turi: [searches Amazon, ranks by Wilson score]
      Here are the top picks under ₹5,000...

You:  Play some lofi music
Turi: [searches YouTube, opens best result in browser]
      Opening lofi hip hop mix...

You:  Remind me to call dentist tomorrow at 9am
Turi: Got it. Remind you to call dentist on [date] at 09:00 AM.
      Confirm? (yes / no)

You:  Open calculator
Turi: Opening calculator.
      [calc.exe launches]

You:  Search for latest news about AI
Turi: [queries SearXNG, synthesizes real results]
      According to recent reports...
```

---

## A Note on the Name

Turi is named after **Alan Mathison Turing** — the man who invented the theoretical foundation of every computer ever built, broke the Nazi Enigma cipher at Bletchley Park, and proposed the Turing Test in 1950 as a measure of machine intelligence.

He was prosecuted for his sexuality in 1952 and received a posthumous royal pardon in 2013. The UK issued a formal apology in 2009.

Every line of code in this system stands on the foundation he built. He deserved better from the world he helped save.

> *"We can only see a short distance ahead, but we can see plenty there that needs to be done."*
> — Alan Turing, 1950

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built with: [llama.cpp](https://github.com/ggerganov/llama.cpp) · [FastAPI](https://fastapi.tiangolo.com) · [ChromaDB](https://www.tricity.dev/chromadb) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [Piper TTS](https://github.com/rhasspy/piper) · [SearXNG](https://github.com/searxng/searxng) · [yt-dlp](https://github.com/yt-dlp/yt-dlp) · [Playwright](https://playwright.dev)

---

*In memory of Alan Turing · 1912–1954*
```