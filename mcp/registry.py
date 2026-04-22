from mcp.base import MCPServer

_servers: dict[str, MCPServer] = {}


def register(server: MCPServer):
    _servers[server.server_name] = server
    print(f"[MCP] registered server: {server.server_name}")


def get(server_name: str) -> MCPServer | None:
    return _servers.get(server_name)


def list_servers() -> list[dict]:
    return [s.tool_schema() for s in _servers.values()]


async def call(
    server_name: str,
    tool_name:   str,
    args:        dict
) -> dict:
    server = _servers.get(server_name)
    if not server:
        return {
            "error": f"MCP server '{server_name}' not registered"
        }
    return await server.call_tool(tool_name, args)