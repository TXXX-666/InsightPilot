$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { throw "Run scripts\setup.ps1 first" }
& $Python -m playwright install chromium
Write-Host "Chromium installed. Dynamic browser_open(render_js=true) is available."

