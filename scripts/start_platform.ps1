<# 
.SYNOPSIS
    Safe one-liner wrapper to start the full platform stack in paper/demo mode:
    - Enforces safety (cwd, no system32, kill stale, KILL_SWITCH check)
    - Starts training loop per symbol (visible in UI)
    - Starts React UI + backend + orchestrator
    - Starts paper trader
    - Enables auto promotion after gates
    - Continuous improvement on new data via supervisor recovery + new training

.DESCRIPTION
    This is the recommended entry point for the verified paper-candidate state.
    Everything stays in paper mode until full gates pass for live.
    UI at http://localhost:5050/ shows per-symbol training performance, equity curves, account info, candidates, champions, safety, decisions with reasons.

.EXAMPLE
    cd C:\supreme-chainsaw; powershell -ExecutionPolicy Bypass -File .\scripts\start_platform.ps1 -Mode paper -Symbols "XAUUSDm,BTCUSDm" -Dashboard -Train -Paper -PromoteAfterGates
#>

[CmdletBinding()]
param(
    [string]$Mode = "paper",  # paper, demo, real (real locked by default)
    [string[]]$Symbols = @("XAUUSDm", "BTCUSDm"),
    [switch]$Dashboard,
    [switch]$Train,
    [switch]$Paper,
    [switch]$PromoteAfterGates,
    [switch]$DryRun,
    [switch]$KillStale,
    [int]$TrainSteps = 50000
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = if ($scriptDir -like "*\scripts") { Split-Path -Parent $scriptDir } else { $scriptDir }
if (-not $RepoRoot -or -not (Test-Path (Join-Path $RepoRoot "launch_full_project.ps1"))) {
    $RepoRoot = (Get-Location).Path
}
Set-Location $RepoRoot

Write-Host "=== start_platform.ps1 - Safe Full Stack Launcher ===" -ForegroundColor Cyan
Write-Host "Repo: $RepoRoot"
Write-Host "Mode: $Mode | Symbols: $($Symbols -join ',')"

# Normalize Symbols if passed as single "XAU,BTC" string (common in one-liners) to array
if ($Symbols.Count -eq 1 -and $Symbols[0] -like '*,*') {
    $Symbols = @($Symbols[0] -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
    Write-Host "Normalized Symbols: $($Symbols -join ',')"
}

# Safety enforcement (from roadmap)
if ($PWD.Path -ne "C:\supreme-chainsaw") {
    Write-Error "Must run from C:\supreme-chainsaw. Current: $PWD"
    exit 1
}
if ($MyInvocation.MyCommand.Path -like "C:\windows\system32*" -or $scriptDir -like "C:\windows\system32*") {
    Write-Error "Cannot run from system32. cd C:\supreme-chainsaw first."
    exit 1
}

# Kill stale / duplicates (always for safety on orchestrator start, per vision; use -KillStale false? but param enables explicit)
# IMPORTANT: never kill self (current $PID) even if cmdline matches
# Legacy names (paper_mt5*, dashboard_backend, live_trade_lane, Server_AGI) are intentionally kept in the match list
# so that any old/outdated processes from previous wiring are forcibly stopped before starting the current safe stack.
if (-not $KillStale.IsPresent -or $KillStale) {
    Write-Host "Killing stale/duplicate processes (safety enforcement)..."
    $currentPid = $PID
    $pyKilled = 0
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessId -ne $currentPid -and $_.CommandLine -and ($_.CommandLine -match 'Server_AGI|api_server|monitor_tui|paper_trader|paper_executor|paper_mt5|run_lane|dashboard_backend|live_trade_lane') } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $pyKilled++ }
    $psKilled = 0
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.ProcessId -ne $currentPid -and $_.CommandLine -and ($_.CommandLine -match 'launch_full_project|vps_agi_supervisor') } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; $psKilled++ }
    Write-Host "  Stopped $pyKilled py + $psKilled ps stale procs (self $currentPid protected)"
    Start-Sleep 3
}

# KILL_SWITCH check (active by default for safety)
$killPath = Join-Path $RepoRoot "runtime\KILL_SWITCH"
if (Test-Path $killPath) {
    Write-Host "KILL_SWITCH active - real execution blocked." -ForegroundColor Yellow
    if ($Mode -eq "real") { $Mode = "paper"; Write-Host "Forcing paper mode." }
}

# Set safe envs
$env:CHAIN_GAMBLER_EXECUTION_MODE = "demo"  # paper/demo
$env:CHAIN_GAMBLER_ALLOW_LIVE = "0"
$env:AGI_AUTO_PROMOTE_CANDIDATE = if ($PromoteAfterGates) { "1" } else { "0" }
$env:AGI_AUTO_MQL5 = if ($PromoteAfterGates) { "1" } else { "0" }
$env:AGI_AUTO_PAPER_HARNESS = if ($Paper) { "1" } else { "0" }
$env:AGI_USE_LEGACY_SINGLE_TF = "0"
$env:AGI_MULTI_TF_STANDARD = "1"
$env:AGI_FEATURE_VERSION = "multitimeframe_best"

Write-Host "Envs set for paper + auto-gates."

# Start training for symbols (background, visible via UI)
if ($Train) {
    Write-Host "Starting training processes..."
    foreach ($sym in $Symbols) {
        $trainCmd = "python -u training/run_lane_b_raw_lstm.py --symbol $sym --steps $TrainSteps --max-dd -40"
        Write-Host "  BG: $trainCmd"
        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "cd '$RepoRoot'; $trainCmd" -WindowStyle Hidden
    }
}

# Start full stack (UI + api_server + supervisor orchestrator)
# Use explicit sub-powershell -File + -Once so it runs to completion (health checks) then exits, leaving bg services; avoids splat/arg parse quirks and blocks until ready
$launchPsArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $RepoRoot 'launch_full_project.ps1'), '-Once')
if ($DryRun) { $launchPsArgs += '-DryRun' }
# Note: -Dashboard means start UI (default in launch); avoid -Preview to use hot-reload dev server on 5173
# If user wants production preview, pass -Preview manually or extend params.
Write-Host "Starting full stack launcher (via sub-powershell -Once; UI+api+supervisor will bg)..."
$lp = Start-Process -FilePath 'powershell.exe' -ArgumentList $launchPsArgs -Wait -NoNewWindow -PassThru
Write-Host "Full stack launcher exited (code: $($lp.ExitCode)) - services should be up."

# Start paper trader for champion trading (if enabled) - uses the paper_trader.py (not execution subpkg)
if ($Paper) {
    Write-Host "Starting paper trader..."
    $symList = ($Symbols -join ' ')
    $paperCmd = "python -m Python.paper_trader --symbols $symList --equity 10000 --no-ollama --cycles 100"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "cd '$RepoRoot'; $paperCmd" -WindowStyle Hidden
}

Write-Host "=== Platform started in $Mode mode ===" -ForegroundColor Green
Write-Host "React UI: http://localhost:5173/ (Vite dev; proxies /api to 5050)"
Write-Host "API (data): http://localhost:5050/api/status (and /api/lanes /api/registry etc for training/equity/trades/safety)"
Write-Host "Supervisor (orchestrator at ~9090) handles recovery + promotion after gates for paper champion only."
Write-Host "Paper trader loads champions per-symbol; training (lane_b + recovery) keeps improving on new data."
Write-Host "Monitor per-symbol progress, candidates, champions, gates, equity, decisions+reasons, L/S/F% etc in the UI."
Write-Host "To stop: use task manager or Ctrl+C in windows; re-run with -KillStale."
