import os
import subprocess
import asyncio
from pathlib import Path
from mcp.base import MCPServer, MCPTool

# read-only allowed extensions
READABLE_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".xlsx",
    ".csv", ".json", ".py", ".js", ".html",
    ".log", ".ini", ".yaml", ".yml"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov",
    ".wmv", ".flv", ".webm", ".m4v"
}

# search roots — customize these
SEARCH_ROOTS = [
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
]


class FileSystemMCPServer(MCPServer):

    @property
    def server_name(self) -> str:
        return "filesystem"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name        = "search_files",
                description = "Search for files and documents on Windows",
                parameters  = {
                    "query":    "string — filename or keyword to search",
                    "filetype": "string — optional: pdf, docx, mp4, etc.",
                }
            ),
            MCPTool(
                name        = "read_file",
                description = "Read content of a text file (read-only)",
                parameters  = {
                    "path": "string — full file path"
                }
            ),
            MCPTool(
                name        = "open_file",
                description = (
                    "Open a file in its default app. "
                    "Videos open in VLC."
                ),
                parameters  = {
                    "path": "string — full file path"
                }
            )
        ]

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name == "search_files":
            return await self._search(args)
        if tool_name == "read_file":
            return await self._read(args)
        if tool_name == "open_file":
            return await self._open(args)
        return {"error": f"unknown tool: {tool_name}"}

    async def _search(self, args: dict) -> dict:
        query    = args.get("query", "").lower()
        filetype = args.get("filetype", "").lower().strip(".")

        print(f"[FS MCP] searching: '{query}' type={filetype}")

        found = []
        for root in SEARCH_ROOTS:
            if not root.exists():
                continue
            try:
                for path in root.rglob("*"):
                    if not path.is_file():
                        continue
                    # match filename
                    if query not in path.name.lower():
                        continue
                    # match filetype if specified
                    if filetype and not path.suffix.lower().endswith(filetype):
                        continue
                    stat = path.stat()
                    found.append({
                        "name":      path.name,
                        "path":      str(path),
                        "extension": path.suffix.lower(),
                        "size_kb":   round(stat.st_size / 1024, 1),
                        "modified":  stat.st_mtime,
                        "readable":  path.suffix.lower() in READABLE_EXTENSIONS,
                        "is_video":  path.suffix.lower() in VIDEO_EXTENSIONS,
                    })
                    if len(found) >= 20:
                        break
            except PermissionError:
                continue

        # sort by modified date — most recent first
        found.sort(key=lambda x: x["modified"], reverse=True)

        print(f"[FS MCP] found {len(found)} files")
        return {
            "status": "ok",
            "query":  query,
            "count":  len(found),
            "files":  found[:10]
        }

    async def _read(self, args: dict) -> dict:
        path = Path(args.get("path", ""))

        if not path.exists():
            return {"status": "error", "error": "file not found"}

        if path.suffix.lower() not in READABLE_EXTENSIONS:
            return {
                "status": "error",
                "error":  f"{path.suffix} files cannot be read as text"
            }

        # hard limit — never read more than 5000 chars
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return {
                "status":    "ok",
                "path":      str(path),
                "content":   content[:5000],
                "truncated": len(content) > 5000,
                "size_chars": len(content)
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def _open(self, args: dict) -> dict:
        path = Path(args.get("path", ""))

        if not path.exists():
            return {"status": "error", "error": "file not found"}

        ext = path.suffix.lower()

        try:
            if ext in VIDEO_EXTENSIONS:
                # try VLC first
                vlc_paths = [
                    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
                    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
                ]
                vlc = next((p for p in vlc_paths if Path(p).exists()), None)

                if vlc:
                    subprocess.Popen([vlc, str(path)])
                    return {
                        "status":  "ok",
                        "message": f"Opening {path.name} in VLC",
                        "app":     "VLC"
                    }
                else:
                    # fallback to default app
                    os.startfile(str(path))
                    return {
                        "status":  "ok",
                        "message": f"Opening {path.name} (VLC not found, using default)",
                        "app":     "default"
                    }
            else:
                # open in default app (read-only intent — Windows handles this)
                os.startfile(str(path))
                return {
                    "status":  "ok",
                    "message": f"Opening {path.name}",
                    "app":     "default"
                }

        except Exception as e:
            return {"status": "error", "error": str(e)}