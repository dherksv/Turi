import os
import re
import subprocess
from pathlib import Path
from mcp.base import MCPServer, MCPTool
from debug_logger import log_event, log_error

READABLE_EXTENSIONS = {
    ".txt", ".md", ".pdf", ".docx", ".xlsx",
    ".csv", ".json", ".py", ".js", ".html",
    ".log", ".ini", ".yaml", ".yml", ".rtf"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".avi", ".mov",
    ".wmv", ".flv", ".webm", ".m4v"
}


def get_search_roots() -> list[Path]:
    """
    Get all valid search roots for this Windows machine.
    Includes the actual user home directory.
    """
    home      = Path.home()
    username  = home.name
    roots     = []

    # standard Windows paths
    candidates = [
        home / "Documents",
        home / "Desktop",
        home / "Downloads",
        home / "OneDrive",
        home / "OneDrive" / "Documents",
        Path(f"C:/Users/{username}/Documents"),
        Path(f"C:/Users/{username}/Desktop"),
        Path(f"C:/Users/{username}/Downloads"),
        # add current working directory too
        Path.cwd(),
        Path.cwd() / "data",
    ]

    for p in candidates:
        if p.exists():
            roots.append(p)
            log_event("fs_root_found", "filesystem",
                      {"path": str(p)})

    print(f"[FS MCP] search roots: "
          f"{[str(r) for r in roots]}")
    return roots


class FileSystemMCPServer(MCPServer):

    @property
    def server_name(self) -> str:
        return "filesystem"

    def list_tools(self) -> list[MCPTool]:
        return [
            MCPTool(
                name        = "search_files",
                description = "Search for files on Windows",
                parameters  = {
                    "query":    "string — filename keyword",
                    "filetype": "string — optional extension"
                }
            ),
            MCPTool(
                name        = "read_file",
                description = "Read text content of a file",
                parameters  = {"path": "string — full file path"}
            ),
            MCPTool(
                name        = "open_file",
                description = "Open file in default app or VLC",
                parameters  = {"path": "string — full file path"}
            ),
            MCPTool(
            name        = "open_app",
            description = "Open any Windows application by name",
            parameters  = {"app": "string — app name like camera, notepad, calculator"}
            ),
             MCPTool(
            name        = "open_explorer",
            description = "Open File Explorer at a specific path",
            parameters  = {"path": "string — folder path (optional)"}
            )
        ]

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name == "search_files":   return await self._search(args)
        if tool_name == "read_file":      return await self._read(args)
        if tool_name == "open_file":      return await self._open(args)
        if tool_name == "open_app":       return await self._open_app(args)
        if tool_name == "open_explorer":  return await self._open_explorer(args)
        return {"error": f"unknown tool: {tool_name}"}

    async def _search(self, args: dict) -> dict:
        raw_query = args.get("query", "").strip()
        filetype  = args.get("filetype", "").lower().strip(".")

        # ── clean the query properly ─────────────────────────
        query = self._clean_query(raw_query)
        if not query and filetype:
            query = ""  # search by type only

        print(f"[FS MCP] raw='{raw_query}' "
              f"cleaned='{query}' type='{filetype}'")

        log_event("fs_search_start", "filesystem", {
            "raw_query": raw_query,
            "clean_query": query,
            "filetype": filetype
        })

        found = []
        roots = get_search_roots()

        for root in roots:
            try:
                for path in root.rglob("*"):
                    if not path.is_file():
                        continue

                    name_lower = path.name.lower()
                    stem_lower = path.stem.lower()

                    # match logic:
                    # 1. if query provided — check if
                    #    ANY query word is in filename
                    # 2. if filetype provided — check extension
                    if query:
                        query_words = [
                            w for w in query.split()
                            if len(w) > 2  # skip tiny words
                        ]
                        if not query_words:
                            # too many words filtered — use raw
                            query_words = query.split()

                        # partial match — any word matches
                        matched = any(
                            w in name_lower or w in stem_lower
                            for w in query_words
                        )
                        if not matched:
                            continue

                    if filetype:
                        if not name_lower.endswith(
                            f".{filetype}"
                        ):
                            continue

                    try:
                        stat = path.stat()
                        found.append({
                            "name":      path.name,
                            "path":      str(path),
                            "extension": path.suffix.lower(),
                            "size_kb":   round(
                                stat.st_size / 1024, 1
                            ),
                            "modified":  stat.st_mtime,
                            "readable":  path.suffix.lower()
                                         in READABLE_EXTENSIONS,
                            "is_video":  path.suffix.lower()
                                         in VIDEO_EXTENSIONS,
                        })
                    except Exception:
                        continue

                    if len(found) >= 30:
                        break

            except PermissionError:
                continue
            except Exception as e:
                log_error(e.__class__.__name__,  e,
                          {"root": str(root)})
                continue

        # sort by most recently modified
        found.sort(
            key     = lambda x: x["modified"],
            reverse = True
        )
        found = found[:10]

        print(f"[FS MCP] found {len(found)} files "
              f"for query='{query}'")

        log_event("fs_search_done", "filesystem", {
            "query":   query,
            "count":   len(found),
            "files":   [f["name"] for f in found]
        })

        if not found:
            # give helpful info about where we searched
            return {
                "status":        "not_found",
                "query":         query,
                "filetype":      filetype,
                "count":         0,
                "files":         [],
                "searched_in":   [str(r) for r in roots],
                "tip": (
                    f"Searched for '{query}' in "
                    f"{len(roots)} locations. "
                    f"Make sure the file is in Documents, "
                    f"Desktop, or Downloads."
                )
            }

        return {
            "status":  "ok",
            "query":   query,
            "count":   len(found),
            "files":   found
        }

    def _clean_query(self, raw: str) -> str:
        """
        Remove command words and keep only the filename/keyword.
        'can you open file Dos_simulation_report.pdf'
        → 'dos simulation report'
        """
        text = raw.lower().strip()

        # remove command phrases
        remove_phrases = [
            "can you open file", "can you open",
            "could you open", "please open",
            "open file", "open the file",
            "open document", "find file",
            "can you find", "can you list",
            "can you read", "please find",
            "search for file", "look for",
            "show me file", "list file",
            "read file", "load file",
        ]
        for phrase in remove_phrases:
            text = text.replace(phrase, "").strip()

        # remove file extension from query
        # (we use filetype param for that)
        text = re.sub(r'\.\w{2,4}$', '', text).strip()

        # remove special chars but keep spaces and hyphens
        text = re.sub(r'[_]', ' ', text)
        text = re.sub(r'[^\w\s\-]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    async def _read(self, args: dict) -> dict:
        path = Path(args.get("path", ""))
        if not path.exists():
            return {"status": "error", "error": "file not found"}
        if path.suffix.lower() not in READABLE_EXTENSIONS:
            return {
                "status": "error",
                "error":  f"cannot read {path.suffix} as text"
            }
        try:
            content = path.read_text(
                encoding="utf-8", errors="ignore"
            )
            return {
                "status":    "ok",
                "path":      str(path),
                "content":   content[:5000],
                "truncated": len(content) > 5000
            }
        except Exception as e:
            log_error("filesystem", e, {"path": str(path)})
            return {"status": "error", "error": str(e)}
# open app
    async def _open_app(self, args: dict) -> dict:
        """Open any Windows application by name."""
        app_name = args.get("app", "").lower().strip()

        # Windows app registry — maps natural names to commands
        APP_MAP = {
            # system apps
            "camera":           "start microsoft.windows.camera:",
            "calculator":       "calc.exe",
            "notepad":          "notepad.exe",
            "paint":            "mspaint.exe",
            "wordpad":          "wordpad.exe",
            "calendar":         "start outlookcal:",
            "mail":             "start outlookmail:",
            "maps":             "start bingmaps:",
            "weather":          "start bingweather:",
            "news":             "start bingnews:",
            "store":            "start ms-windows-store:",
            "settings":         "start ms-settings:",
            "clock":            "start ms-clock:",
            "alarms":           "start ms-clock:",
            "photos":           "start ms-photos:",
            "movies":           "start mswindowsvideo:",
            "groove":           "start mswindowsmusic:",
            "music":            "start mswindowsmusic:",
            "xbox":             "start xbox:",
            "cortana":          "start cortana:",
            "edge":             "start microsoft-edge:",
            "browser":          "start microsoft-edge:",

            # file system
            "explorer":         "explorer.exe",
            "file explorer":    "explorer.exe",
            "files":            "explorer.exe",

            # office
            "word":             "winword.exe",
            "excel":            "excel.exe",
            "powerpoint":       "powerpnt.exe",
            "outlook":          "outlook.exe",
            "onenote":          "onenote.exe",
            "teams":            "teams.exe",

            # dev tools
            "vscode":           "code.exe",
            "vs code":          "code.exe",
            "visual studio code": "code.exe",
            "terminal":         "wt.exe",
            "powershell":       "powershell.exe",
            "cmd":              "cmd.exe",
            "command prompt":   "cmd.exe",
            "task manager":     "taskmgr.exe",
            "device manager":   "devmgmt.msc",
            "control panel":    "control.exe",
            "registry":         "regedit.exe",

            # media
            "vlc":              "vlc.exe",
            "spotify":          "spotify.exe",
            "discord":          "discord.exe",
            "zoom":             "zoom.exe",
            "obs":              "obs64.exe",
            "steam":            "steam.exe",

            # utilities
            "snipping tool":    "snippingtool.exe",
            "snip":             "snippingtool.exe",
            "screenshot":       "snippingtool.exe",
            "magnifier":        "magnify.exe",
            "narrator":         "narrator.exe",
            "on screen keyboard": "osk.exe",
            "character map":    "charmap.exe",
            "paint 3d":         "paint3d.exe",
            "3d viewer":        "start 3dviewer:",
            "sticky notes":     "start ms-stickynotes:",
            "whiteboard":       "start ms-whiteboard:",
        }

        # fuzzy match — find best app
        matched_cmd  = None
        matched_name = None

        # exact match first
        if app_name in APP_MAP:
            matched_cmd  = APP_MAP[app_name]
            matched_name = app_name
        else:
            # partial match
            for key, cmd in APP_MAP.items():
                if app_name in key or key in app_name:
                    matched_cmd  = cmd
                    matched_name = key
                    break

        if not matched_cmd:
            # try running directly as executable
            matched_cmd  = f"{app_name}.exe"
            matched_name = app_name

        print(f"[APP] opening '{matched_name}' → {matched_cmd}")
        log_event("app_open", "filesystem", {
            "requested": app_name,
            "matched":   matched_name,
            "command":   matched_cmd
        })

        try:
            import subprocess
            if matched_cmd.startswith("start "):
                # URI scheme — use shell=True
                subprocess.Popen(
                    matched_cmd,
                    shell  = True,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL
                )
            else:
                subprocess.Popen(
                    matched_cmd,
                    shell  = True,
                    stdout = subprocess.DEVNULL,
                    stderr = subprocess.DEVNULL
                )

            return {
                "status":  "ok",
                "message": f"Opening {matched_name}",
                "app":     matched_name,
                "command": matched_cmd
            }

        except Exception as e:
            log_error("filesystem", e, {"app": app_name})
            return {
                "status":  "error",
                "error":   str(e),
                "message": f"Could not open {app_name}"
            }

    
    async def _open(self, args: dict) -> dict:
        path = Path(args.get("path", ""))
        if not path.exists():
            return {
                "status": "error",
                "error":  f"file not found: {path}"
            }

        ext = path.suffix.lower()
        log_event("fs_open", "filesystem", {
            "path": str(path),
            "ext":  ext
        })

        try:
            if ext in VIDEO_EXTENSIONS:
                vlc_paths = [
                    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
                    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
                ]
                vlc = next(
                    (p for p in vlc_paths if Path(p).exists()),
                    None
                )
                if vlc:
                    import subprocess
                    subprocess.Popen([vlc, str(path)])
                    return {
                        "status":  "ok",
                        "message": f"Opening {path.name} in VLC",
                        "app":     "VLC",
                        "path":    str(path)
                    }

            # default Windows open
            os.startfile(str(path))
            return {
                "status":  "ok",
                "message": f"Opening {path.name}",
                "app":     "default",
                "path":    str(path)
            }
        except Exception as e:
            log_error("filesystem", e, {"path": str(path)})
            return {"status": "error", "error": str(e)}
        
    async def _open_explorer(self, args: dict) -> dict:
        """Open File Explorer at specific path."""
        import subprocess
        path = args.get("path", "")
        try:
            if path and Path(path).exists():
                subprocess.Popen(["explorer.exe", path])
                return {
                    "status":  "ok",
                    "message": f"Opened File Explorer at {path}"
                }
            else:
                subprocess.Popen(["explorer.exe"])
                return {
                    "status":  "ok",
                    "message": "Opened File Explorer"
                }
        except Exception as e:
            return {"status": "error", "error": str(e)}    