$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\setup.ps1 first" }
. (Join-Path $PSScriptRoot "start-common.ps1")

if (Test-PortInUse -Port 8000) {
    if (Test-InsightPilotApi) {
        Write-Host "InsightPilot API is already running at http://127.0.0.1:8000" -ForegroundColor Green
        return
    }
    throw "Port 8000 is used by another application. Stop it or change the InsightPilot API port."
}
Set-Location $ProjectRoot
& $Python -m uvicorn insightpilot.api:app --app-dir "$ProjectRoot\python" --host 127.0.0.1 --port 8000
