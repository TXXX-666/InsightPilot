$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\setup.ps1 first" }
Set-Location $ProjectRoot
& $Python -m uvicorn insightpilot.api:app --app-dir "$ProjectRoot\python" --host 127.0.0.1 --port 8000

