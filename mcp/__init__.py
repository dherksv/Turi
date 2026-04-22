from mcp.registry      import register, call, list_servers, get
from mcp.amazon_server  import AmazonMCPServer
from mcp.youtube_server import YouTubeMCPServer
from mcp.filesystem_server import FileSystemMCPServer

def init_mcp():
    """Register all MCP servers."""
    register(AmazonMCPServer())
    register(YouTubeMCPServer())
    register(FileSystemMCPServer())
    print("[MCP] all servers registered")