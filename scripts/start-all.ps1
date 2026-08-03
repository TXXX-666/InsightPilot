$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) { throw "Run scripts\setup.ps1 first" }
Start-Process powershell -WindowStyle Hidden -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$ProjectRoot\scripts\start-api.ps1`""
Start-Sleep -Seconds 2
& "$ProjectRoot\.venv\Scripts\streamlit.exe" run "$ProjectRoot\frontend\app.py" --server.address 127.0.0.1 --server.port 8501

