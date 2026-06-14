<# 
.SYNOPSIS
    Supreme Chainsaw - Launch Full Production Monitoring Stack (One-Command)

.DESCRIPTION
    Robust launcher for the full observable production monitoring stack on Windows VPS.
    Starts the React UI (frontend/), the dedicated api_server (5050) that powers its data,
    the vps_agi_supervisor (core trading/equity/training recovery/candidate orchestration),
    and optional TUI (with parity features).

    NEW STANDARD (default, no overrides): full multi-timeframe pipeline (1m+5m+15m+1h per symbol)
    using best known feature params from configs/best_features_per_symbol.yaml (auto via multitimeframe_builder).
    Training recovery, arming, and execution handoff all default to fetch_multitimeframe + best params.

    Strong Node detection for shells without PATH. Dry-run, health checks, logging, graceful shutdown.
    Legacy single-TF: AGI_USE_LEGACY_SINGLE_TF=1 (preserved).

    Default: React dev server (hot reload + proxy to 5050 for complete data) + updated TUI.
    -Preview: production build preview mode.

    Ports: 5173 (UI), 5050 (api data), 9090 (Server_AGI via supervisor).

.EXAMPLE
    .\launch_full_project.ps1                 # Full stack (recommended)
.EXAMPLE
    .\launch_full_project.ps1 -Preview -DryRun
.EXAMPLE
    .\launch_full_project.ps1 -NoSupervisor -NoTui
#>

[CmdletBinding()]
param(
    [switch]$Preview,
    [switch]$NoBuild,
    [switch]$NoSupervisor,
    [switch]$NoTui,
    [switch]$NoBrowser,
    [int]$UiPort = 5173,
    [int]$ApiPort = 5050,
    [string]$HealthBaseServer = "http://127.0.0.1:9090",
    [switch]$KillStale,
    [switch]$DryRun,
    [switch]$Once,
    [int]$HealthPollSeconds = 30
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSCommandPath
if (-not $RepoRoot) { $RepoRoot = (Get-Location).Path }
Set-Location $RepoRoot

# Log setup and helpers MUST be defined before first Write-LaunchLog (was bug: call before def + before $LaunchLog)
$LogsDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$LaunchLog = Join-Path $LogsDir "launch_full_project.log"
$TmpDir = Join-Path $RepoRoot ".tmp"
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

$script:ChildPids = [System.Collections.Generic.List[int]]::new()

function Write-LaunchLog {
    param([string]$Level = "INFO", [string]$Message)
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss.fff")
    $line = "[$ts] [$Level] $Message"
    Add-Content -Path $LaunchLog -Value $line -Encoding UTF8 -Force -ErrorAction SilentlyContinue
    $color = switch ($Level) { "ERROR" {"Red"} "WARN" {"Yellow"} "SUCCESS" {"Green"} default {"Cyan"} }
    Write-Host $line -ForegroundColor $color
}

function Rotate-LogIfLarge { param([string]$Path, [int]$MaxMB = 10)
    if ((Test-Path $Path) -and ((Get-Item $Path).Length -gt ($MaxMB * 1MB))) {
        $bak = "$Path.1"; Remove-Item $bak -Force -ErrorAction SilentlyContinue
        Move-Item $Path $bak -Force -ErrorAction SilentlyContinue
    }
}
Rotate-LogIfLarge -Path $LaunchLog

# ============================================================
# NEW MULTI-TIMEFRAME STANDARD (1m + 5m + 15m + 1h per symbol)
# Uses configs/best_features_per_symbol.yaml loaded via Python/features/multitimeframe_builder.py
# + fetch_multitimeframe_training_data + build_multitimeframe_feature_matrix
# This is now the easy default for full stack, training recovery, and handoff arming.
# TUI with full React parity features + React UI started by default.
# Legacy single-TF fallback preserved: set AGI_USE_LEGACY_SINGLE_TF=1 (or per-call overrides)
# ============================================================
$env:AGI_USE_LEGACY_SINGLE_TF = "0"
$env:AGI_MULTI_TF_STANDARD = "1"
$env:AGI_FEATURE_VERSION = "multitimeframe_best"
$env:AGI_MTF_TIMEFRAMES = "1m,5m,15m,1h"
Write-LaunchLog "INFO" "NEW STANDARD active: AGI_MULTI_TF_STANDARD=1 (best features per symbol; legacy via AGI_USE_LEGACY_SINGLE_TF=1)"

Write-LaunchLog "INFO" "=== Launch Full Project starting (DryRun=$DryRun, Preview=$Preview) ==="
Write-LaunchLog "INFO" "RepoRoot: $RepoRoot"

Write-Host ""
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host "  Supreme Chainsaw - Full Production Monitoring Stack Launcher  " -ForegroundColor Cyan
Write-Host "================================================================" -ForegroundColor Cyan
Write-Host " React UI (frontend/) + api_server (5050) + Supervisor (9090) + TUI (parity)" -ForegroundColor Green
Write-Host " NEW STANDARD DEFAULT: 1m+5m+15m+1h multi-TF + best_features_per_symbol.yaml via multitimeframe_builder" -ForegroundColor Magenta
Write-Host "   (Legacy single-TF: set AGI_USE_LEGACY_SINGLE_TF=1 before launch)" -ForegroundColor DarkGray
Write-Host ""

if ($DryRun) {
    Write-Host "[DRY-RUN] Plan only - no processes started." -ForegroundColor Yellow
}

# Python detection (exact match to existing project patterns)
function Find-Python {
    $cands = @(
        (Join-Path $RepoRoot ".venv312\Scripts\python.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe")
    )
    foreach ($c in $cands) { if (Test-Path $c) { Write-LaunchLog "SUCCESS" "Python: $c"; return $c } }
    Write-LaunchLog "ERROR" "No venv python (.venv312 or .venv). Run: python -m venv .venv312 then pip install -r requirements.txt"
    return $null
}
$PythonExe = Find-Python
if (-not $PythonExe -and -not $DryRun) { exit 1 }

# Robust Node detection (handles non-PATH VPS shells)
function Find-NodeAndNpm {
    $nodeExe = $null; $npmCmd = $null

    try {
        $nc = Get-Command node -ErrorAction SilentlyContinue
        if ($nc) { $nodeExe = $nc.Source; $nm = Get-Command npm -ErrorAction SilentlyContinue; if ($nm) { $npmCmd = $nm.Source } }
    } catch {}

    if (-not $nodeExe) {
        $paths = @(
            (Join-Path $env:ProgramFiles 'nodejs\node.exe'),
            (Join-Path ${env:ProgramFiles(x86)} 'nodejs\node.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\nodejs\node.exe'),
            (Join-Path $env:LOCALAPPDATA 'Programs\Node.js\node.exe'),
            ($env:USERPROFILE + '\AppData\Roaming\nvm\nodejs\node.exe'),
            'C:\Program Files\nodejs\node.exe'
        )
        foreach ($p in $paths) { if (Test-Path $p) { $nodeExe = $p; break } }
    }

    if (-not $nodeExe) {
        $roots = @($env:ProgramFiles, ${env:ProgramFiles(x86)}, $env:LOCALAPPDATA)
        foreach ($r in $roots) {
            if (-not $r -or -not (Test-Path $r)) { continue }
            try {
                $h = Get-ChildItem -Path $r -Recurse -Filter 'node.exe' -ErrorAction SilentlyContinue -Depth 4 |
                     Where-Object { $_.FullName -match 'nodejs|node\\node' } | Select-Object -First 1
                if ($h) { $nodeExe = $h.FullName; break }
            } catch {}
        }
    }

    if ($nodeExe -and (Test-Path $nodeExe)) {
        $nodeDir = Split-Path $nodeExe -Parent
        $ncands = @((Join-Path $nodeDir 'npm.cmd'), (Join-Path $nodeDir 'npm.exe'))
        foreach ($n in $ncands) { if (Test-Path $n) { $npmCmd = $n; break } }

        if ($nodeDir -and ($env:PATH -notlike ('*' + $nodeDir + '*'))) {
            $env:PATH = $nodeDir + ';' + $env:PATH
            Write-LaunchLog 'INFO' ('Node PATH injected: ' + $nodeDir)
        }
        Write-LaunchLog 'SUCCESS' ('Node: ' + $nodeExe + ' npm: ' + $npmCmd)
        return @{Node=$nodeExe; Npm=$npmCmd; Dir=$nodeDir}
    }
    Write-LaunchLog 'ERROR' 'Node/npm not found. Install Node 18+ from nodejs.org'
    return $null
}

$NodeInfo = Find-NodeAndNpm
if (-not $NodeInfo) {
    if (-not $DryRun) { exit 1 }
    $NpmExe = $null
    $NodeExe = $null
} else {
    $NpmExe = $NodeInfo.Npm
    $NodeExe = $NodeInfo.Node
}

if ($NodeInfo -and -not $DryRun) {
    try {
        $v = & $NodeExe --version 2>$null
        Write-LaunchLog 'INFO' ('Node version: ' + $v)
    } catch {}
}

# Kill stale
if ($KillStale) {
    Write-LaunchLog 'INFO' 'KillStale: cleaning prior processes'
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'Server_AGI|api_server|monitor_tui' } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Start-Sleep 2
}

if ($DryRun) {
    Write-Host 'DRY-RUN PLAN:' -ForegroundColor Yellow
    Write-Host ('  Python : ' + $PythonExe)
    Write-Host ('  Node   : ' + $NodeExe)
    Write-Host ('  npm    : ' + $NpmExe)
    Write-Host ('  api_server on ' + $ApiPort)
    Write-Host ('  supervisor (unless -NoSupervisor)')
    $m = if ($Preview) { 'preview (build)' } else { 'dev (hot reload + proxy)' }
    Write-Host ('  React ' + $m + ' on ' + $UiPort)
    Write-Host '  Health checks + optional browser + TUI'
    Write-LaunchLog 'INFO' 'DRY-RUN complete'
    exit 0
}

# Tracked process starter
function Start-Tracked {
    param($FilePath, $ArgumentList, $WorkingDirectory = $RepoRoot, $LogFile, $Title = '', [switch]$NoWindow, [switch]$Min)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $FilePath
    $psi.Arguments = ($ArgumentList -join ' ')
    $psi.WorkingDirectory = $WorkingDirectory
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $NoWindow.IsPresent
    if ($Min) { $psi.WindowStyle = [System.Diagnostics.ProcessWindowStyle]::Minimized }
    if ($LogFile) { $psi.RedirectStandardOutput = $true; $psi.RedirectStandardError = $true }
    try {
        $p = [System.Diagnostics.Process]::Start($psi)
        if ($p) {
            $script:ChildPids.Add($p.Id) | Out-Null
            if ($LogFile) {
                # fire and forget log tailers
                Start-Job -ScriptBlock { param($proc,$logf) while (-not $proc.HasExited) { $l = $proc.StandardOutput.ReadLine(); if ($l) { Add-Content $logf $l } } } -ArgumentList $p,$LogFile | Out-Null
                Start-Job -ScriptBlock { param($proc,$logf) while (-not $proc.HasExited) { $l = $proc.StandardError.ReadLine(); if ($l) { Add-Content $logf ('[E] ' + $l) } } } -ArgumentList $p,$LogFile | Out-Null
            }
            Write-LaunchLog 'SUCCESS' ("Started $Title PID=" + $p.Id)
            return $p.Id
        }
    } catch { Write-LaunchLog 'ERROR' ("Start failed for $Title : " + $_) }
    return $null
}

function Stop-Tracked {
    foreach ($id in $script:ChildPids) {
        try { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue } catch {}
    }
    $script:ChildPids.Clear()
}

# Cleanup registration
try { Register-EngineEvent PowerShell.Exiting -Action { Stop-Tracked } -ErrorAction SilentlyContinue | Out-Null } catch {}

# 1. API server (React data layer)
$apiLog = Join-Path $LogsDir 'api_server_full.log'
Write-LaunchLog 'INFO' ("Starting api_server on $ApiPort (React monitoring data)")
Start-Tracked -FilePath $PythonExe -ArgumentList @('-m','Python.api_server') -LogFile $apiLog -Title 'api_server-5050' -NoWindow
Start-Sleep 5

# 2. Supervisor (inherits AGI_MULTI_TF_STANDARD + best-features envs set above for new pipeline defaults)
$supPid = $null
if (-not $NoSupervisor) {
    $supPath = Join-Path $RepoRoot 'scripts\vps_agi_supervisor.ps1'
    if (Test-Path $supPath) {
        Write-LaunchLog 'INFO' 'Starting vps_agi_supervisor (background) [NEW MTF STANDARD env inherited]'
        $supPid = Start-Tracked -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$supPath`"") -LogFile (Join-Path $LogsDir 'supervisor_via_launcher.log') -Title 'supervisor' -NoWindow
        Start-Sleep 6
    } else {
        $ss = Join-Path $RepoRoot 'start_server.ps1'
        if (Test-Path $ss) {
            Start-Tracked -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$ss`"") -Title 'Server-AGI' -NoWindow
        }
    }
}

# 3. React UI
$feDir = Join-Path $RepoRoot 'frontend'
if (Test-Path $feDir) {
    if (-not $NoBuild) {
        Push-Location $feDir
        try {
            if ($NpmExe) { & $NpmExe install --prefer-offline --no-audit --no-fund 2>&1 | Out-Null } else { npm install --prefer-offline --no-audit --no-fund 2>&1 | Out-Null }
        } catch {}
        if ($Preview) {
            try { if ($NpmExe) { & $NpmExe run build 2>&1 | Out-Null } else { npm run build 2>&1 | Out-Null } } catch {}
        }
        Pop-Location
    }
    $uiLog = Join-Path $LogsDir 'react_ui.log'
    $args = if ($Preview) { @('run','preview','--','--port',$UiPort,'--host','0.0.0.0') } else { @('run','dev','--','--port',$UiPort,'--host','0.0.0.0') }
    Push-Location $feDir
    Start-Tracked -FilePath $NpmExe -ArgumentList $args -WorkingDirectory $feDir -LogFile $uiLog -Title 'React-UI' -Min
    Pop-Location
    Start-Sleep 8
}

# 4. Optional TUI (updated with full React parity features for pipeline/brains/gates/equity/decisions + MTF awareness)
if (-not $NoTui) {
    $tuiL = Join-Path $RepoRoot 'launch_tui.ps1'
    if (Test-Path $tuiL) {
        Start-Tracked -FilePath 'powershell.exe' -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',"`"$tuiL`"",'-Watcher','-Persistent') -LogFile (Join-Path $LogsDir 'tui_via_launcher.log') -Title 'TUI (parity+MTF)' -NoWindow
    }
}

# Health waits
function Test-Url { param($u, $t=6) try { return (Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec $t -ErrorAction Stop).StatusCode -eq 200 } catch { return $false } }

Write-LaunchLog 'INFO' 'Health checks...'
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
    $a = Test-Url "http://127.0.0.1:$ApiPort/api/status"
    $s = Test-Url "$HealthBaseServer/api/health"
    $u = Test-Url "http://127.0.0.1:$UiPort/"
    if ($a -and $s -and $u) { Write-LaunchLog 'SUCCESS' 'All core services healthy'; break }
    Start-Sleep 3
}

# Browser
if (-not $NoBrowser) {
    $url = "http://localhost:$UiPort/"
    Write-LaunchLog 'INFO' ("Opening React UI: $url")
    Start-Process $url | Out-Null
}

# Banner
Write-Host ""
Write-Host "================================================================" -ForegroundColor Green
Write-Host "  FULL STACK LAUNCHED - React + TUI (parity) | NEW MTF STANDARD (1m+5m+15m+1h + best feats)" -ForegroundColor Green
Write-Host "================================================================" -ForegroundColor Green
Write-Host " UI: http://localhost:$UiPort/" -ForegroundColor White
Write-Host " API: http://localhost:$ApiPort/api/status" -ForegroundColor White
Write-Host " Health: $HealthBaseServer/api/health" -ForegroundColor White
Write-Host " Log: $LaunchLog" -ForegroundColor DarkGray
Write-Host " Env: AGI_MULTI_TF_STANDARD=1 (override legacy with AGI_USE_LEGACY_SINGLE_TF=1)" -ForegroundColor DarkGray
Write-Host ""

# Status loop or exit
if (-not $Once) {
    Write-LaunchLog 'INFO' 'Status loop running (Ctrl+C to stop)'
    try {
        while ($true) {
            Start-Sleep $HealthPollSeconds
            $ok = (Test-Url "http://127.0.0.1:$ApiPort/api/status" 5) -and (Test-Url "$HealthBaseServer/api/health" 5)
            Write-LaunchLog ('INFO') ('Health tick: API+Server OK=' + $ok)
        }
    } finally {
        Stop-Tracked
        Write-LaunchLog 'SUCCESS' 'Shutdown complete'
    }
} else {
    Write-LaunchLog 'INFO' 'Once mode - background services left running'
}

exit 0
