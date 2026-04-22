from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPTool:
    name:        str
    description: str
    parameters:  dict   # JSON schema of expected args


class MCPServer(ABC):
    """
    Base class for all MCP tool servers.
    Each server registers tools and handles calls.
    """

    @property
    @abstractmethod
    def server_name(self) -> str:
        pass

    @abstractmethod
    def list_tools(self) -> list[MCPTool]:
        pass

    @abstractmethod
    async def call_tool(
        self,
        tool_name: str,
        args:      dict
    ) -> dict:
        pass

    def tool_schema(self) -> dict:
        """Returns full server schema — used by orchestrator discovery."""
        return {
            "server":  self.server_name,
            "tools":   [
                {
                    "name":        t.name,
                    "description": t.description,
                    "parameters":  t.parameters
                }
                for t in self.list_tools()
            ]
        }