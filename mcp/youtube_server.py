import asyncio
import json
import subprocess
from mcp.base import MCPServer, MCPTool


class YouTubeMCPServer(MCPServer):

    @property
    def server_name(self) -> str:
        return "youtube"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name        = "search_videos",
                description = "Search YouTube for videos or music",
                parameters  = {
                    "query":       "string — search query",
                    "max_results": "number — how many results (default 5)",
                    "type":        "string — 'music' or 'video' (default 'video')"
                }
            ),
            MCPTool(
                name        = "get_stream_url",
                description = "Get direct stream URL for a YouTube video",
                parameters  = {
                    "video_url": "string — full YouTube URL",
                    "quality":   "string — 'audio' or 'video' (default 'audio')"
                }
            )
        ]

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name == "search_videos":
            return await self._search(args)
        if tool_name == "get_stream_url":
            return await self._get_stream(args)
        return {"error": f"unknown tool: {tool_name}"}

    async def _search(self, args: dict) -> dict:
        query       = args.get("query", "")
        max_results = int(args.get("max_results", 5))
        media_type  = args.get("type", "video")

        # add music keywords if music requested
        if media_type == "music":
            query = f"{query} music"

        print(f"[YOUTUBE MCP] searching: '{query}'")

        try:
            # yt-dlp search — no API key needed
            cmd = [
                "yt-dlp",
                f"ytsearch{max_results}:{query}",
                "--dump-json",
                "--flat-playlist",
                "--no-warnings",
                "--quiet"
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(
                result.communicate(), timeout=30
            )

            videos = []
            for line in stdout.decode().strip().split('\n'):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    videos.append({
                        "id":          data.get("id", ""),
                        "title":       data.get("title", "")[:100],
                        "url":         f"https://youtube.com/watch?v={data.get('id','')}",
                        "duration":    data.get("duration"),
                        "view_count":  data.get("view_count", 0),
                        "uploader":    data.get("uploader", ""),
                        "thumbnail":   data.get("thumbnail", ""),
                        "description": (data.get("description") or "")[:200],
                    })
                except json.JSONDecodeError:
                    continue

            print(f"[YOUTUBE MCP] found {len(videos)} videos")
            return {
                "status":  "ok",
                "query":   query,
                "type":    media_type,
                "count":   len(videos),
                "videos":  videos
            }

        except asyncio.TimeoutError:
            return {"status": "error", "error": "search timed out", "videos": []}
        except FileNotFoundError:
            return {"status": "error",
                    "error": "yt-dlp not installed — run: pip install yt-dlp",
                    "videos": []}
        except Exception as e:
            return {"status": "error", "error": str(e), "videos": []}

    async def _get_stream(self, args: dict) -> dict:
        video_url = args.get("video_url", "")
        quality   = args.get("quality", "audio")

        print(f"[YOUTUBE MCP] getting stream: {video_url} ({quality})")

        try:
            if quality == "audio":
                fmt = "bestaudio[ext=m4a]/bestaudio/best"
            else:
                fmt = "best[height<=720]/best"

            cmd = [
                "yt-dlp",
                "--get-url",
                "--format", fmt,
                "--no-warnings",
                "--quiet",
                video_url
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE
            )
            stdout, _ = await asyncio.wait_for(
                result.communicate(), timeout=30
            )

            stream_url = stdout.decode().strip()
            if not stream_url:
                return {"status": "error", "error": "no stream URL found"}

            return {
                "status":     "ok",
                "stream_url": stream_url,
                "video_url":  video_url,
                "quality":    quality
            }

        except Exception as e:
            return {"status": "error", "error": str(e)}