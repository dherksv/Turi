"""
Human Evaluation Script
Presents test prompts and collects human ratings.
Saves structured results for academic reporting.
"""

import asyncio
import json
import time
import httpx
import sys
sys.path.insert(0, '..')

BASE_URL = "http://localhost:3001"

EVAL_PROMPTS = [
    {
        "id":       "Q1",
        "category": "Factual",
        "prompt":   "What is the capital of France?",
        "criteria": "accuracy"
    },
    {
        "id":       "Q2",
        "category": "Tool — Web Search",
        "prompt":   "Search for latest news about artificial intelligence",
        "criteria": "tool_use + relevance"
    },
    {
        "id":       "Q3",
        "category": "Tool — Shopping",
        "prompt":   "Find wireless headset under 3000 rupees",
        "criteria": "tool_use + filter_accuracy"
    },
    {
        "id":       "Q4",
        "category": "Tool — YouTube",
        "prompt":   "Play a lofi music video",
        "criteria": "tool_use + correct_media_type"
    },
    {
        "id":       "Q5",
        "category": "Reasoning",
        "prompt":   "Compare Python and JavaScript for building AI applications",
        "criteria": "depth + accuracy + relevance"
    },
    {
        "id":       "Q6",
        "category": "Scheduling",
        "prompt":   "Remind me to drink water in 2 minutes",
        "criteria": "tool_use + time_parse + confirmation"
    },
    {
        "id":       "Q7",
        "category": "Chitchat",
        "prompt":   "Hello, how are you?",
        "criteria": "naturalness"
    },
    {
        "id":       "Q8",
        "category": "Contextual",
        "prompt":   "What is your name and who are you named after?",
        "criteria": "identity + Turing knowledge"
    },
    {
        "id":       "Q9",
        "category": "Complex Reasoning",
        "prompt":   "What are the pros and cons of a multi-agent AI system versus a single model assistant?",
        "criteria": "depth + accuracy + structure"
    },
    {
        "id":       "Q10",
        "category": "Indirect Command",
        "prompt":   "Can you open the calculator for me?",
        "criteria": "intent_recognition + tool_use"
    },
]


async def get_response(session_id: str, message: str) -> tuple[str, int]:
    start      = time.time()
    full_reply = ""

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={"session_id": session_id, "message": message}
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                    if event.get("done"):
                        full_reply = event.get("full", "")
                        break
                except Exception:
                    continue

    latency_ms = int((time.time() - start) * 1000)
    return full_reply, latency_ms


async def run_human_eval():
    import os
    os.makedirs("results", exist_ok=True)

    async with httpx.AsyncClient() as client:
        resp       = await client.post(f"{BASE_URL}/session/new")
        session_id = resp.json()["session_id"]

    print(f"\n{'═'*65}")
    print(f"  TURI SYSTEM — HUMAN EVALUATION")
    print(f"  Rate each response 1–5 on the given criteria")
    print(f"  1=Poor  2=Fair  3=Good  4=Very Good  5=Excellent")
    print(f"{'═'*65}")

    all_scores = []

    for item in EVAL_PROMPTS:
        print(f"\n{'─'*65}")
        print(f"  [{item['id']}] {item['category']}")
        print(f"  Criteria: {item['criteria']}")
        print(f"  Prompt: {item['prompt']}")
        print(f"{'─'*65}")
        print(f"  Getting response...")

        reply, latency = await get_response(session_id, item["prompt"])

        print(f"\n  RESPONSE ({latency}ms):")
        print(f"  {reply[:400]}")
        if len(reply) > 400:
            print(f"  ... [{len(reply)-400} more chars]")

        print(f"\n  Rate this response (1-5): ", end="")
        while True:
            try:
                score = int(input().strip())
                if 1 <= score <= 5:
                    break
                print("  Enter 1-5: ", end="")
            except ValueError:
                print("  Enter 1-5: ", end="")

        print(f"  Notes (optional, press Enter to skip): ", end="")
        notes = input().strip()

        all_scores.append({
            "id":         item["id"],
            "category":   item["category"],
            "prompt":     item["prompt"],
            "criteria":   item["criteria"],
            "reply":      reply[:500],
            "latency_ms": latency,
            "score":      score,
            "notes":      notes
        })

        print(f"  Recorded: {score}/5")

    # summary
    scores     = [r["score"] for r in all_scores]
    avg        = sum(scores) / len(scores)
    by_cat     = {}
    for r in all_scores:
        cat = r["category"].split("—")[0].strip()
        by_cat.setdefault(cat, []).append(r["score"])

    print(f"\n{'═'*65}")
    print(f"  EVALUATION SUMMARY")
    print(f"{'─'*65}")
    print(f"  Overall average score : {avg:.2f} / 5.00")
    print(f"\n  By category:")
    for cat, cat_scores in by_cat.items():
        cat_avg = sum(cat_scores) / len(cat_scores)
        print(f"    {cat:<25} {cat_avg:.2f}/5")

    report = {
        "session_id":    session_id,
        "total_prompts": len(EVAL_PROMPTS),
        "overall_avg":   round(avg, 3),
        "by_category":   {
            cat: round(sum(s)/len(s), 3)
            for cat, s in by_cat.items()
        },
        "responses":     all_scores
    }

    with open("results/human_eval_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Full results saved to results/human_eval_results.json")
    return report


if __name__ == "__main__":
    asyncio.run(run_human_eval())