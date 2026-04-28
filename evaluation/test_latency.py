"""
Latency Benchmark
Measures response time for fast path vs deep path.
Tests tool execution time separately.
"""

import asyncio
import time
import json
import sys
import statistics
sys.path.insert(0, '..')

import httpx

BASE_URL = "http://localhost:3001"

TEST_QUERIES = {
    "fast_path": [
        "what time is it",
        "play lofi music",
        "open calculator",
        "hello",
        "search for python tutorials",
    ],
    "deep_path": [
        "explain how transformers work in machine learning",
        "suggest the best programming language for AI development",
        "compare python and javascript for backend development",
        "help me plan a study schedule for next week",
        "what are the pros and cons of microservices architecture",
    ],
    "tool_calls": [
        "search for latest news about artificial intelligence",
        "find wireless headset under 3000 rupees",
        "put on a lofi music video",
    ]
}


async def measure_latency(
    session_id: str,
    message:    str
) -> dict:
    start = time.time()
    first_chunk_time = None
    full_reply = []

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={
                "session_id": session_id,
                "message":    message
            }
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    import json as j
                    event = j.loads(line[6:])

                    if event.get("chunk") and first_chunk_time is None:
                        first_chunk_time = time.time() - start

                    if event.get("done"):
                        full_reply = event.get("full", "")
                        break
                except Exception:
                    continue

    total_time = time.time() - start

    return {
        "message":          message[:50],
        "total_ms":         round(total_time * 1000),
        "first_chunk_ms":   round((first_chunk_time or total_time) * 1000),
        "reply_length":     len(full_reply)
    }


async def run_benchmark():
    # create session
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}/session/new")
        session_id = resp.json()["session_id"]

    print(f"Session: {session_id[:8]}")
    print(f"\nWarming up...")

    # warmup
    await measure_latency(session_id, "hello")
    await asyncio.sleep(1)

    all_results = {}

    for category, queries in TEST_QUERIES.items():
        print(f"\n{'═'*60}")
        print(f"  {category.upper().replace('_',' ')}")
        print(f"{'═'*60}")

        timings = []
        first_chunk_timings = []

        for query in queries:
            result = await measure_latency(session_id, query)
            timings.append(result["total_ms"])
            first_chunk_timings.append(result["first_chunk_ms"])

            print(
                f"  [{result['total_ms']:>5}ms total | "
                f"{result['first_chunk_ms']:>4}ms first chunk] "
                f"{query[:45]}"
            )
            await asyncio.sleep(0.5)

        avg_total       = statistics.mean(timings)
        avg_first       = statistics.mean(first_chunk_timings)
        median_total    = statistics.median(timings)

        print(f"\n  Average total    : {avg_total:.0f}ms")
        print(f"  Average 1st chunk: {avg_first:.0f}ms")
        print(f"  Median total     : {median_total:.0f}ms")
        print(f"  Min / Max        : {min(timings)}ms / {max(timings)}ms")

        all_results[category] = {
            "avg_total_ms":       round(avg_total),
            "avg_first_chunk_ms": round(avg_first),
            "median_ms":          round(median_total),
            "min_ms":             min(timings),
            "max_ms":             max(timings),
            "samples":            len(timings)
        }

    # save
    with open("results/latency_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to results/latency_results.json")

    # summary table
    print(f"\n{'SUMMARY':^60}")
    print(f"{'Category':<20} {'Avg Total':>12} {'Avg 1st Chunk':>14}")
    print("─" * 50)
    for cat, r in all_results.items():
        print(
            f"{cat:<20} "
            f"{r['avg_total_ms']:>10}ms "
            f"{r['avg_first_chunk_ms']:>12}ms"
        )


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    asyncio.run(run_benchmark())