import json
import asyncio
import subprocess
import webbrowser
from mcp.base import MCPServer, MCPTool
from debug_logger import log_event, log_error


class YouTubeMCPServer(MCPServer):

    @property
    def server_name(self) -> str:
        return "youtube"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name        = "search_videos",
                description = "Search YouTube",
                parameters  = {
                    "query":       "string",
                    "max_results": "number",
                    "type":        "string — music or video"
                }
            ),
            MCPTool(
                name        = "open_video",
                description = "Open a YouTube URL in browser",
                parameters  = {"url": "string"}
            )
        ]

    async def call_tool(
        self, tool_name: str, args: dict
    ) -> dict:
        if tool_name == "search_videos":
            return await self._search(args)
        if tool_name == "open_video":
            return self._open_in_browser(args)
        return {"error": f"unknown tool: {tool_name}"}

    def _open_in_browser(self, args: dict) -> dict:
        url = args.get("url", "")
        if not url:
            return {"status": "error", "error": "no url"}
        try:
            webbrowser.open(url)
            log_event("youtube_opened", "youtube_mcp",
                      {"url": url})
            return {
                "status":  "ok",
                "message": f"Opened in browser: {url}",
                "url":     url
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _search(self, args: dict) -> dict:
        query       = args.get("query", "")
        max_results = int(args.get("max_results", 5))
        media_type  = args.get("type", "video")

        # clean query
        import re
        clean_query = re.sub(
            r'^(play|watch|stream|listen\s+to|'
            r'put\s+(on|a|an)?|show\s+me|'
            r'find|search\s+for)\s+',
            '', query, flags=re.IGNORECASE
        ).strip()

        if media_type == "music":
            search_q = f"{clean_query} music"
        else:
            search_q = clean_query

        print(f"[YOUTUBE MCP] searching: '{search_q}'")
        log_event("youtube_search_start", "youtube_mcp", {
            "raw_query":   query,
            "clean_query": search_q,
            "type":        media_type
        })

        try:
            result = await asyncio.to_thread(
                self._search_sync, search_q, max_results
            )
            result["media_type"] = media_type
            result["clean_query"] = clean_query

            # auto-open best result in browser
            if (result.get("status") == "ok"
                    and result.get("videos")):
                best_url = result["videos"][0]["url"]
                webbrowser.open(best_url)
                result["opened_url"] = best_url
                print(f"[YOUTUBE MCP] opened: {best_url}")
                log_event("youtube_auto_opened",
                          "youtube_mcp",
                          {"url": best_url})

            return result

        except Exception as e:
            log_error("youtube_mcp", e, {"query": search_q})
            return {
                "status": "error",
                "error":  f"{type(e).__name__}: {str(e)}",
                "videos": []
            }

    def _search_sync(
        self, query: str, max_results: int
    ) -> dict:
        print(f"[YOUTUBE SYNC] running yt-dlp for '{query}'")

        cmd = [
            "yt-dlp",
            f"ytsearch{max_results}:{query}",
            "--dump-json",
            "--flat-playlist",
            "--no-warnings",
            "--quiet"
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output = True,
                timeout        = 30,
                text           = True
            )

            if result.returncode != 0 and result.stderr:
                print(f"[YOUTUBE SYNC] yt-dlp stderr: "
                      f"{result.stderr[:200]}")

            videos = []
            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    vid_id = data.get("id", "")
                    videos.append({
                        "id":        vid_id,
                        "title":     data.get("title", "")[:100],
                        "url":       (
                            f"https://youtube.com/watch?v={vid_id}"
                        ),
                        "duration":  data.get("duration"),
                        "view_count": data.get("view_count", 0),
                        "uploader":  data.get("uploader", ""),
                        "thumbnail": data.get("thumbnail", ""),
                    })
                except json.JSONDecodeError:
                    continue

            print(f"[YOUTUBE SYNC] found {len(videos)} videos")

            log_event("youtube_search_done", "youtube_mcp", {
                "query":  query,
                "count":  len(videos),
                "titles": [v["title"][:40]
                           for v in videos[:3]]
            })

            return {
                "status":  "ok",
                "query":   query,
                "count":   len(videos),
                "videos":  videos
            }

        except FileNotFoundError:
            msg = ("yt-dlp not installed. "
                   "Run: pip install yt-dlp")
            print(f"[YOUTUBE SYNC] {msg}")
            return {
                "status": "error",
                "error":  msg,
                "videos": []
            }
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error":  "yt-dlp search timed out",
                "videos": []
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"[YOUTUBE SYNC] error: {e}\n{tb}")
            return {
                "status": "error",
                "error":  str(e),
                "videos": []
            }