$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Streamlit = Join-Path $ProjectRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $Streamlit)) { throw "Run scripts\setup.ps1 first" }
Set-Location $ProjectRoot
& $Streamlit run "$ProjectRoot\frontend\app.py" --server.address 127.0.0.1 --server.port 8501

