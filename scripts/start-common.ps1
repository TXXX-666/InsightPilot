function Test-PortInUse {
    param([Parameter(Mandatory = $true)][int]$Port)

    return ([System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()).Port -contains $Port
}

function Get-ListeningProcessId {
    param([Parameter(Mandatory = $true)][int]$Port)

    $Pattern = "^\s*TCP\s+127\.0\.0\.1:$Port\s+.*\s+(\d+)\s*$"
    foreach ($Line in & "$env:SystemRoot\System32\netstat.exe" -ano -p TCP) {
        $Match = [regex]::Match($Line, $Pattern)
        if ($Match.Success) {
            return [int]$Match.Groups[1].Value
        }
    }
    return $null
}

function Test-InsightPilotApi {
    try {
        $Schema = Invoke-RestMethod -Uri "http://127.0.0.1:8000/openapi.json" -TimeoutSec 3
        return $Schema.info.title -eq "InsightPilot API"
    }
    catch {
        return $false
    }
}

function Test-InsightPilotUi {
    param([Parameter(Mandatory = $true)][string]$ProjectRoot)

    try {
        $Health = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8501/_stcore/health" -TimeoutSec 3
        if ($Health.StatusCode -ne 200 -or $Health.Content.Trim() -ne "ok") {
            return $false
        }

        $OwningProcess = Get-ListeningProcessId -Port 8501
        if (-not $OwningProcess) {
            return $false
        }
        $Process = Get-WmiObject Win32_Process -Filter "ProcessId=$OwningProcess" -ErrorAction Stop
        $ExpectedApp = (Join-Path $ProjectRoot "frontend\app.py").ToLowerInvariant()
        return $Process.CommandLine.ToLowerInvariant().Contains($ExpectedApp)
    }
    catch {
        return $false
    }
}

function Wait-InsightPilotApi {
    param([int]$TimeoutSeconds = 20)

    $Deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $Deadline) {
        if (Test-InsightPilotApi) {
            return $true
        }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Write-InsightPilotUrls {
    Write-Host "InsightPilot is already running." -ForegroundColor Green
    Write-Host "UI:  http://127.0.0.1:8501"
    Write-Host "API: http://127.0.0.1:8000/docs"
}
