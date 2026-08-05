$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Streamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $Streamlit)) { throw "Run scripts\setup.ps1 first" }
. (Join-Path $PSScriptRoot "start-common.ps1")

if (Test-PortInUse -Port 8501) {
    if (Test-InsightPilotUi -ProjectRoot $ProjectRoot) {
        Write-Host "InsightPilot UI is already running at http://127.0.0.1:8501" -ForegroundColor Green
        return
    }
    throw "Port 8501 is used by another application. Stop it or change the InsightPilot UI port."
}
Set-Location $ProjectRoot
& $Streamlit run "$ProjectRoot\frontend\app.py" --server.address 127.0.0.1 --server.port 8501
