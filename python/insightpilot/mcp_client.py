"""MCP client with stdio and Streamable HTTP transports.

The official MCP Python SDK is imported lazily, so the core REST product keeps
working when the optional SDK is not installed or no MCP server is configured.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ENV_REF = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


class McpDependencyMissing(RuntimeError):
    pass


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_REF.sub(lambda match: os.getenv(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _safe_endpoint(url: str) -> str:
    """Return only scheme and hostname. Hosted MCP URLs may embed credentials."""
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.hostname:
        return "configured"
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}/***"


def _safe_error(exc: Exception, config: dict[str, Any]) -> str:
    message = str(exc) or type(exc).__name__
    url = str(config.get("url") or "")
    if url:
        message = message.replace(url, _safe_endpoint(url))
        parsed = urlparse(url)
        if parsed.path and parsed.path != "/":
            message = message.replace(parsed.path, "/***")
        if parsed.query:
            message = message.replace(parsed.query, "***")
    return message


def _content_to_text(result: Any) -> str:
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
            continue
        if hasattr(block, "model_dump"):
            parts.append(json.dumps(block.model_dump(mode="json"), ensure_ascii=False))
        else:
            parts.append(str(block))
    structured = getattr(result, "structuredContent", None)
    if structured is None:
        structured = getattr(result, "structured_content", None)
    if structured is not None:
        parts.append(json.dumps(structured, ensure_ascii=False))
    return "\n".join(parts) or "(empty MCP result)"


class McpConnection:
    def __init__(self, server_name: str, config: dict[str, Any]):
        self.server_name = server_name
        self.config = config
        self.transport = self._transport_type(config)
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self.tools: list[dict[str, Any]] = []
        self.error: str | None = None

    @staticmethod
    def _transport_type(config: dict[str, Any]) -> str:
        declared = str(config.get("type") or config.get("transport") or "").lower()
        if declared in {"streamable-http", "streamable_http", "http", "remote"} or config.get("url"):
            return "streamable-http"
        return "stdio"

    @property
    def endpoint(self) -> str | None:
        url = str(self.config.get("url") or "")
        return _safe_endpoint(url) if url else None

    async def connect(self) -> None:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:
            raise McpDependencyMissing(
                "MCP Python SDK 未安装。运行 scripts/install-mcp.ps1"
            ) from exc

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        try:
            if self.transport == "streamable-http":
                url = str(self.config.get("url") or "").strip()
                if not url:
                    raise ValueError("Streamable HTTP MCP URL 未配置")
                kwargs: dict[str, Any] = {}
                signature = inspect.signature(streamable_http_client)
                headers = self.config.get("headers") or {}
                if headers and "http_client" in signature.parameters:
                    import httpx
                    http_client = await self._stack.enter_async_context(
                        httpx.AsyncClient(headers={str(k): str(v) for k, v in headers.items()})
                    )
                    kwargs["http_client"] = http_client
                transport_result = await self._stack.enter_async_context(
                    streamable_http_client(url, **kwargs)
                )
            else:
                command = str(self.config.get("command") or "").strip()
                if not command:
                    raise ValueError("stdio MCP command 未配置")
                params = StdioServerParameters(
                    command=command,
                    args=[str(arg) for arg in self.config.get("args", [])],
                    env={**os.environ, **{str(k): str(v) for k, v in (self.config.get("env") or {}).items()}},
                )
                transport_result = await self._stack.enter_async_context(stdio_client(params))

            read_stream, write_stream = transport_result[0], transport_result[1]
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except Exception:
            await self.close()
            raise

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._session is None:
            return []
        result = await self._session.list_tools()
        self.tools = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema or {"type": "object", "properties": {}},
                "serverName": self.server_name,
            }
            for tool in result.tools
        ]
        return self.tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        if self._session is None:
            raise RuntimeError(f"MCP server '{self.server_name}' is not connected")
        result = await self._session.call_tool(name, arguments=args)
        return _content_to_text(result)

    async def close(self) -> None:
        if self._stack is not None:
            try:
                await self._stack.aclose()
            except (Exception, asyncio.CancelledError):
                pass
        self._stack = None
        self._session = None


class McpManager:
    def __init__(self):
        self._connections: dict[str, McpConnection] = {}
        self._tools: list[dict[str, Any]] = []
        self._configured: dict[str, dict[str, Any]] = {}
        self._errors: dict[str, str] = {}
        self._connected = False

    async def load_and_connect(self) -> None:
        if self._connected:
            return
        self._connected = True
        self._configured = self._load_configs()
        if not self._configured:
            return

        timeout = float(os.getenv("INSIGHTPILOT_MCP_CONNECT_TIMEOUT", "20"))
        for name, config in self._configured.items():
            connection = McpConnection(name, config)
            try:
                await asyncio.wait_for(connection.connect(), timeout=timeout)
                tools = await asyncio.wait_for(connection.list_tools(), timeout=timeout)
                self._connections[name] = connection
                self._tools.extend(tools)
                print(f"[mcp] Connected to '{name}' ({connection.transport}) - {len(tools)} tools", flush=True)
            except Exception as exc:
                message = _safe_error(exc, config)
                self._errors[name] = message
                print(f"[mcp] Failed to connect to '{name}': {message}", flush=True)
                await connection.close()

    def get_tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": f"mcp__{tool['serverName']}__{tool['name']}",
                "description": tool.get("description") or f"MCP tool {tool['name']} from {tool['serverName']}",
                "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for tool in self._tools
        ]

    def tools_for_server(self, server_name_contains: str) -> list[dict[str, Any]]:
        """Return discovered tools for connected servers matching a stable label.

        Hosted MCP endpoints can use different server names, so product adapters
        match a descriptive fragment (for example ``arxiv``) instead of relying
        on a credential-bearing URL or a single vendor-specific name.
        """
        needle = server_name_contains.casefold().strip()
        return [
            {
                **tool,
                "prefixedName": f"mcp__{tool['serverName']}__{tool['name']}",
            }
            for tool in self._tools
            if needle in str(tool.get("serverName", "")).casefold()
        ]

    def is_mcp_tool(self, name: str) -> bool:
        return name.startswith("mcp__")

    async def call_tool(self, prefixed_name: str, args: dict[str, Any]) -> str:
        parts = prefixed_name.split("__")
        if len(parts) < 3:
            raise ValueError(f"Invalid MCP tool name: {prefixed_name}")
        server_name = parts[1]
        tool_name = "__".join(parts[2:])
        connection = self._connections.get(server_name)
        if connection is None:
            raise RuntimeError(f"MCP server '{server_name}' not connected")
        try:
            return await connection.call_tool(tool_name, args)
        except Exception as exc:
            raise RuntimeError(self.redact_error(exc)) from exc

    def redact_error(self, exc: BaseException) -> str:
        message = str(exc) or type(exc).__name__
        for config in self._configured.values():
            message = _safe_error(RuntimeError(message), config)
        return message

    def status(self) -> dict[str, Any]:
        servers: list[dict[str, Any]] = []
        for name, config in self._configured.items():
            connection = self._connections.get(name) or McpConnection(name, config)
            servers.append(
                {
                    "name": name,
                    "transport": connection.transport,
                    "endpoint": connection.endpoint,
                    "status": "connected" if name in self._connections else "error",
                    "tool_count": len(connection.tools) if name in self._connections else 0,
                    "error": self._errors.get(name),
                }
            )
        return {
            "configured": len(self._configured),
            "connected": len(self._connections),
            "tool_count": len(self._tools),
            "servers": servers,
        }

    async def disconnect_all(self) -> None:
        await asyncio.gather(
            *(connection.close() for connection in self._connections.values()),
            return_exceptions=True,
        )
        self._connections.clear()
        self._tools.clear()
        self._connected = False

    def _load_configs(self) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        self._merge_config_file(Path.home() / ".insightpilot" / "settings.json", merged)
        self._merge_config_file(Path.cwd() / ".insightpilot" / "settings.json", merged)
        self._merge_config_file(Path.cwd() / ".mcp.json", merged)

        hosted_tavily = os.getenv("MODELSCOPE_TAVILY_MCP_URL", "").strip()
        if hosted_tavily and "modelscope-tavily" not in merged:
            merged["modelscope-tavily"] = {
                "type": "streamable-http",
                "url": hosted_tavily,
            }
        hosted_arxiv = os.getenv("MODELSCOPE_ARXIV_MCP_URL", "").strip()
        if hosted_arxiv and "modelscope-arxiv" not in merged:
            merged["modelscope-arxiv"] = {
                "type": "streamable-http",
                "url": hosted_arxiv,
            }
        return merged

    def _merge_config_file(self, path: Path, target: dict[str, dict[str, Any]]) -> None:
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            servers = raw.get("mcpServers", raw)
            for name, original in servers.items():
                if not isinstance(original, dict):
                    continue
                config = _expand_env(original)
                has_stdio = bool(config.get("command"))
                has_remote = bool(config.get("url"))
                if has_stdio or has_remote:
                    target[str(name)] = config
        except Exception as exc:
            self._errors[f"config:{path.name}"] = str(exc)
