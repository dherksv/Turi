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
    """
    Query SearXNG and return cleaned results.
    Returns a dict with 'results' list and 'status'.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{SEARXNG_URL}/search",
                params={
                    "q":        query,
                    "format":   "json",
                    "language": language,
                    "safesearch": "0",
                    "categories": "general"
                }
            )
            resp.raise_for_status()
            data = resp.json()

        raw_results = data.get("results", [])[:num_results]

        # clean and normalize each result
        results = []
        for r in raw_results:
            results.append({
                "title":   r.get("title", "").strip(),
                "url":     r.get("url", ""),
                "snippet": _clean_snippet(r.get("content", "")),
                "engine":  r.get("engine", "")
            })

        return {
            "status":  "ok",
            "query":   query,
            "count":   len(results),
            "results": results
        }

    except httpx.TimeoutException:
        return {
            "status":  "error",
            "query":   query,
            "error":   "search timed out",
            "results": []
        }
    except httpx.ConnectError:
        return {
            "status":  "error",
            "query":   query,
            "error":   "SearXNG not reachable — is Docker running?",
            "results": []
        }
    except Exception as e:
        return {
            "status":  "error",
            "query":   query,
            "error":   str(e),
            "results": []
        }


def _clean_snippet(text: str) -> str:
    """Remove HTML tags and excess whitespace from snippets."""
    import re
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:300]  # cap at 300 chars per snippet


def results_to_context(search_result: dict) -> str:
    """
    Convert search results into a compact string
    for injection into the LLM context.
    """
    if search_result["status"] == "error":
        return f"Web search failed: {search_result['error']}"

    if not search_result["results"]:
        return f"No results found for: {search_result['query']}"

    lines = [f"Web search results for '{search_result['query']}':"]
    for i, r in enumerate(search_result["results"], 1):
        lines.append(f"\n[{i}] {r['title']}")
        lines.append(f"    {r['snippet']}")
        lines.append(f"    Source: {r['url']}")

    return "\n".join(lines)