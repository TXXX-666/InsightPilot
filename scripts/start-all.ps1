$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) { throw "Run scripts\setup.ps1 first" }
. (Join-Path $PSScriptRoot "start-common.ps1")

$ApiRunning = Test-InsightPilotApi
$UiRunning = Test-InsightPilotUi -ProjectRoot $ProjectRoot

if ((Test-PortInUse -Port 8000) -and -not $ApiRunning) {
    throw "Port 8000 is used by another application. Stop it or change the InsightPilot API port."
}
if ((Test-PortInUse -Port 8501) -and -not $UiRunning) {
    throw "Port 8501 is used by another application. Stop it or change the InsightPilot UI port."
}
if ($ApiRunning -and $UiRunning) {
    Write-InsightPilotUrls
    return
}

if (-not $ApiRunning) {
    Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$ProjectRoot\scripts\start-api.ps1`""
    if (-not (Wait-InsightPilotApi)) {
        throw "InsightPilot API did not become ready within 20 seconds. Check the API process output."
    }
}

if ($UiRunning) {
    Write-InsightPilotUrls
    return
}

& "$ProjectRoot\.venv\Scripts\streamlit.exe" run "$ProjectRoot\frontend\app.py" --server.address 127.0.0.1 --server.port 8501
