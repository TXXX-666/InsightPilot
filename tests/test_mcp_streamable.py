from __future__ import annotations

import asyncio
import socket

import pytest
import uvicorn
from mcp.server.fastmcp import FastMCP

from insightpilot.mcp_client import McpConnection


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_streamable_http_handshake_discovery_and_call():
    port = free_port()
    server = FastMCP("insightpilot-test", stateless_http=True, json_response=True)

    @server.tool()
    def echo(text: str) -> str:
        """Return text for MCP client integration testing."""
        return f"echo:{text}"

    uvicorn_server = uvicorn.Server(
        uvicorn.Config(server.streamable_http_app(), host="127.0.0.1", port=port, log_level="error")
    )
    server_task = asyncio.create_task(uvicorn_server.serve())
    try:
        for _ in range(100):
            if uvicorn_server.started:
                break
            await asyncio.sleep(0.02)
        assert uvicorn_server.started

        connection = McpConnection(
            "test-remote",
            {"type": "streamable-http", "url": f"http://127.0.0.1:{port}/mcp"},
        )
        await connection.connect()
        tools = await connection.list_tools()
        assert [tool["name"] for tool in tools] == ["echo"]
        result = await connection.call_tool("echo", {"text": "hello"})
        assert "echo:hello" in result
        await connection.close()
    finally:
        uvicorn_server.should_exit = True
        await server_task

