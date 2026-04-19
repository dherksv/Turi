import re
import httpx
import os
from dotenv import load_dotenv

load_dotenv()
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8888")


async def search_web(
    query:       str,
    num_results: int = 5,
    language:    str = "en"
) -> dict:

    print(f"\n[SEARCH] query='{query}' → {SEARXNG_URL}")

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q":          query,
                    "format":     "json",
                    "language":   language,
                    "safesearch": "0",
                    "categories": "general"
                }
            )
            print(f"[SEARCH] HTTP status: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()

        raw     = data.get("results", [])[:num_results]
        print(f"[SEARCH] results returned: {len(raw)}")

        if not raw:
            print(f"[SEARCH] WARNING — SearXNG returned 0 results")
            print(f"[SEARCH] full response keys: {list(data.keys())}")

        results = []
        for r in raw:
            results.append({
                "title":   r.get("title",   "").strip(),
                "url":     r.get("url",     ""),
                "snippet": _clean_snippet(r.get("content", "")),
                "engine":  r.get("engine",  "")
            })
            print(f"[SEARCH]   → {r.get('title','')[:60]}")

        return {
            "status":  "ok",
            "query":   query,
            "count":   len(results),
            "results": results
        }

    except httpx.TimeoutException:
        print(f"[SEARCH] ERROR — timed out")
        return {
            "status":  "error",
            "query":   query,
            "error":   "search timed out",
            "results": []
        }
    except httpx.ConnectError as e:
        print(f"[SEARCH] ERROR — cannot connect: {e}")
        return {
            "status":  "error",
            "query":   query,
            "error":   "SearXNG not reachable — is Docker running?",
            "results": []
        }
    except Exception as e:
        print(f"[SEARCH] ERROR — {type(e).__name__}: {e}")
        return {
            "status":  "error",
            "query":   query,
            "error":   str(e),
            "results": []
        }


def _clean_snippet(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:300]


def results_to_context(search_result: dict) -> str:
    if search_result["status"] == "error":
        return f"Web search failed: {search_result.get('error', 'unknown')}"
    if not search_result["results"]:
        return f"No results found for: {search_result['query']}"
    lines = [f"Web search results for '{search_result['query']}':"]
    for i, r in enumerate(search_result["results"], 1):
        lines.append(f"\n[{i}] {r['title']}")
        lines.append(f"    {r['snippet']}")
        lines.append(f"    Source: {r['url']}")
    return "\n".join(lines)