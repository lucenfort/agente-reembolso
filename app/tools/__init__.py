"""Ferramentas e integrações externas."""

from .mcp_client import MCPClient, obter_cliente_mcp

__all__ = [
    "MCPClient",
    "obter_cliente_mcp",
]
