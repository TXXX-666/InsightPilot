$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\setup.ps1 first" }
& $Python -m pip install "mcp>=1.29.0,<2.0.0"
& $Python -c "from mcp import ClientSession; from mcp.client.streamable_http import streamable_http_client; print('MCP SDK ready')"
