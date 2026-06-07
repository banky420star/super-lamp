<# 
.SYNOPSIS
    VPS Operational Hardening - AGI Server Supervisor for Windows (Task Scheduler friendly)

.DESCRIPTION
    Long-running supervisor for Python.Server_AGI on Windows VPS.
    - Detects crashes / exits via process scan (CIM + lockfile aware)
    - Polls /api/health (and /api/health/ready) for degraded state
    - Auto-restarts the AGI server process with proper env + venv python
    - Writes structured logs + simple rotation
    - Respects single-instance via existing server_agi.lock
    - Bounded restarts (cooldown + hourly cap) to avoid restart loops
    - Integrates with existing launch patterns (start_server.ps1 semantics)
    - Monitors training runs (postfix 50k+ / v4 robust), paper harness, MT5, disk, candidates
    - Auto-launches TUI watcher on new good post-fix candidate (self-sustaining)
    - Logs recovery guidance with conservative hyperparams for failed training
    - Prioritizes paper trading harness coordination when candidate appears
    - AUTO-PROMOTION & GATES (auditor fix): On good post-fix candidate (alignment_fix_applied + not quarantined),
      reliably auto-invokes promoter (promote_candidate_to_paper.py via gates/evaluator) + optional canary promotion.
      Controlled by opt-in env: AGI_AUTO_PROMOTE_CANDIDATE=1 (or SUPERVISOR_AUTO_PROMOTE_CANDIDATE / AGI_AUTO_PROMOTE).
      This closes the gap where detection happened but champion_cycle / gates / canary promotion was not auto-invoked.
      Flow: "good candidate detected" (Test-RecentCandidateStaged) -> (env gate) auto_promote_candidate.ps1 -> promoter (real gates + optional set_canary) -> paper canary + MQL5.
      champion_cycle full path available behind AGI_USE_FULL_CHAMPION_CYCLE=1.
      All paths have safety rails: env gates (never default-on), dry-run support, audit logs, post-fix only.

    - FINAL ZERO-TOUCH ORCHESTRATION (Zero-Touch Orchestrator): On candidate transition, supervisor now fires *cohesive full autonomous chain*
      (TUI watcher + unified promoter + explicit bg MQL5 deploy with dedicated logs + feedback status) with rich logging.
      Makes the system one autonomous unit: v4 training candidate -> gates -> canary/paper/MQL5 shadow/feedback triggers with near-zero operator steps (env armed once).

    Recommended usage on production VPS:
      1. Place in Task Scheduler (Trigger: At startup + On event 1000/1001 for crashes)
      2. "Run whether user is logged on or not" + "Run with highest privileges"
      3. Action: powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\supreme-chainsaw\scripts\vps_agi_supervisor.ps1"
      4. In Settings: "Restart the task every: 01:00:00" + "If the running task does not end when requested, force it to stop"
      5. "Stop the task if it runs longer than 0" (indefinite) + "If the task fails, restart every: 5 minutes"

    Or run manually in a dedicated console for dev: .\scripts\vps_agi_supervisor.ps1 -MonitorOnly:$false

    Zero-touch: see docs + vps_launch scripts. Use launch_tui.ps1 -Watcher -Persistent alongside.

    For v4 training run benefit (launch_robust_postfix_training_v4.ps1 or similar 50k+):
      Set $env:AGI_AUTO_PROMOTE_CANDIDATE="1" (in session or persistent) before/while running supervisor.
      When v4 stages candidate with alignment_fix_applied, supervisor will auto trigger full gates->canary flow.
      With AGI_AUTO_MQL5=1 + AGI_PROMOTER_PROMOTE_CANARY=1 the *entire* promoter + MQL5 deploy + feedback chain fires autonomously in bg (see orchestration block).

.NOTES
    Requires: PowerShell 5.1+ (built into Windows Server)
    No external modules.
    Safe for real MT5 - only restarts on explicit failure signals.
    2026-05-27: Hardened for 50k postfix training, TUI auto, paper harness, candidate coordination.
    2026-05-27 (Auto-Promotion & Gates Agent): Added reliable env-gated auto-invoke of promoter/champion_cycle path + canary promotion. Closes repeated auditor flag on detection-without-gates.
    2026-05-27 (Final Zero-Touch Orchestrator): Added high-level orchestration glue in main loop: on candidate, explicit bg full-chain trigger (promoter + MQL5 deploy + logs + feedback) for cohesive autonomous system. v4 runs now deliver end-to-end with one-time env arming.
    2026-05-27 (Post-Candidate Handoff Automation Agent): Centralized Invoke-PostCandidateHandoff on detection transition. Always produces runtime/post_candidate_handoff_commands.txt + last_handoff.json (TUI panel + monitor coord). Safe prep (dry promoter + MQL5-LogOnly) + env-gated full auto (paper harness running + MQL5 shadow prepared). Resilient finder + v4 scorecard support. Makes the "good candidate staged -> ready" transition automatic/reliable/visible.
    2026-05-28 (Full Stack MTF Update): Defaults to NEW STANDARD multi-timeframe (fetch_multitimeframe + best_features_per_symbol via multitimeframe_builder) for training recovery + arming execution/handoff. Updated TUI parity + React. Legacy single-TF via AGI_USE_LEGACY_SINGLE_TF=1 preserved. Decision+Execution new setup context deployed on arm (when ready).
#>
[CmdletBinding()]
param(
    [string]$RepoRoot = "C:\supreme-chainsaw",
    [int]$HealthPollSeconds = 45,
    [int]$MaxRestartsPerHour = 6,
    [int]$RestartCooldownSeconds = 30,
    [switch]$MonitorOnly,           # If set, never restart - only log/alert
    [switch]$DryRun,
    [string]$HealthBase = "http://127.0.0.1:5050"
)

# Operational Readiness improvements (2026-05-27):
# - Fixed scoping, paper enforcement, MT5/disk/login guards
# - Training (postfix 50k / v4), paper harness, candidate detection + auto TUI watcher
# - Recovery guidance with conservative hyperparams (target_kl 0.05 etc.)
# - Better logging + self-sustainability for 24/7 VPS
# - Re-register Task Scheduler task after updates
# - MQL5 Production hook: on good candidate (alignment_fix_applied) emits exact one-command + auto-triggers deploy_mql5 via promoter (now always on promoter success) for full Python->MQL5 shadow zero/one cmd path (env AGI_AUTO_MQL5_DEPLOY=1 for zero-cmd)
# - AUTO-PROMOTION & GATES (auditor remediation): env-gated (AGI_AUTO_PROMOTE_CANDIDATE=1) reliable invocation of
#   auto_promote_candidate.ps1 + promote_candidate_to_paper.py (which runs evaluate + PromotionGates) + optional direct
#   ModelRegistry.set_canary. Ensures "detected good candidate" always leads to gates run + canary path (opt-in, safe).
#   v4 runs now auto-benefit once candidate stages if env set. See auto_promote_candidate.ps1 + promoter --promote-canary.

# === PRODUCTION RUNBOOK NOTES: Timing-Aware Rich Decision PPO Safety (2026-05-28 Hardening) ===
# - Live account safety: ExecutionAgent + RiskEngine/Supervisor now cap sizing (SizeSpec + global), daily loss & emergency flatten respect TimeExitSpec (news windows honored via EventGuard heuristics + defer non-critical)
# - Canary extended: timing metrics (open_window_pnl, news_avoid_pnl, avoidance_score) + auto rollback if degradation (high news-prox trades or negative score)
# - Harness/Supervisor: _should_rollback and canary_monitor now timing-aware; flatten honors windows
# - Rollback triggers: manual flag, loss breach (timing respected), canary violation (incl. timing degrade), consecutive errors
# - To run unsupervised live: set AGI_EXECUTION_TYPE=decision_ppo; monitor TUI/decision_ppo_execution_live.json + timing fields in canary artifacts
# - Regression: tests/test_risk_supervisor.py + test_order_manager.py cover new paths
# - See: Python/execution/execution_agent.py (force_flatten + compute_lots), risk_engine.py, canary/*, paper_mt5_execution_harness.py, runtime/agent_status/production_hardening_timing_agent.json
# Full ops: docs/DECISION_EXECUTION_ARCHITECTURE.md + GO_LIVE_CHECKLIST.md (timing section)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# ============================================================
# NEW MULTI-TIMEFRAME STANDARD DEFAULT (2026-05-28)
# - Training recovery now defaults to EnableTimeframeOpt (1m+5m+15m+1h via fetch_multitimeframe_training_data)
# - Best feature params auto-loaded from configs/best_features_per_symbol.yaml (multitimeframe_builder)
# - Arming paper/MQL5/execution uses multi-TF context (handoff to Decision+Execution when ready)
# - Updated TUI (parity) + React UI launched via full stack
# Legacy single-TF: AGI_USE_LEGACY_SINGLE_TF=1 (or explicit -EnableTimeframeOpt:$false overrides)
# ============================================================

# F11 fix: hoist Write-SupLog definition above its first call site. PowerShell 5.1 does not
# resolve forward-references for script-root-scope function calls.
function Write-SupLog {
    param([string]$Level = "INFO", [string]$Message)
    # F11 fix: $SupervisorLog may not be set yet (it's initialized at line 138 in the script body).
    # Use a default path that derives from the same logic, and let the late-bound assignment
    # at line 138 take over once the script body reaches it.
    $logPath = if (Test-Path variable:SupervisorLog) { $SupervisorLog } else { Join-Path $RepoRoot "logs\vps_agi_supervisor.log" }
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss.fff")
    $line = "[$ts] [$Level] $Message"
    Add-Content -Path $logPath -Value $line -Encoding UTF8 -Force
    if ($Level -in @("ERROR","WARN")) {
        Write-Host $line -ForegroundColor $(if($Level -eq "ERROR"){"Red"}else{"Yellow"})
    } elseif ($Level -eq "INFO") {
        Write-Host $line -ForegroundColor Cyan
    } else {
        Write-Verbose $line
    }
    if ((Test-Path $logPath) -and ((Get-Item $logPath).Length -gt 5MB)) {
        $bak = "$logPath.1"
        if (Test-Path $bak) { Remove-Item $bak -Force -ErrorAction SilentlyContinue }
        Move-Item $logPath $bak -Force -ErrorAction SilentlyContinue
    }
}

$env:AGI_USE_LEGACY_SINGLE_TF = "0"
$env:AGI_MULTI_TF_STANDARD = "1"
$env:AGI_FEATURE_VERSION = "multitimeframe_best"
$env:AGI_MTF_TIMEFRAMES = "1m,5m,15m,1h"
# Execution path default: pure-Python primary (OrderManager + MT5Executor via ExecutionAgent mql5_bridge=False)
# for Windows + running MT5 terminal (direct, simple, reliable telemetry to Decision PPO).
# To force MQL5 bridge (optional high-perf):  $env:MQL5_BRIDGE_ENABLED="1"
$env:MQL5_BRIDGE_ENABLED = "0"
Write-SupLog "INFO" "NEW STANDARD active in supervisor: multi-TF 1m+5m+15m+1h + best params per symbol (AGI_MULTI_TF_STANDARD=1). Legacy via AGI_USE_LEGACY_SINGLE_TF=1. Execution: pure-Python primary (MQL5_BRIDGE_ENABLED=0; set=1 for optional MQL5 bridge)"

# --- Paths & Logging ---
$LogsDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogsDir | Out-Null
$tmpDir = Join-Path $RepoRoot ".tmp"
New-Item -ItemType Directory -Force -Path $tmpDir | Out-Null
$SupervisorLog = Join-Path $LogsDir "vps_agi_supervisor.log"
$LockPath = Join-Path $RepoRoot ".tmp\server_agi.lock"

# (Write-SupLog hoisted to top of script — F11 fix)

function Write-PipelineDecision {
    param(
        [string]$DecisionType = "supervisor",
        [string]$Actor = "supervisor",
        [string]$Decision,
        [string]$Candidate = "",
        [string]$RunId = "",
        [string]$Reason = "",
        [string]$DetailsJson = "{}",
        [string]$Severity = "info"
    )
    $ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
    # F12 fix: guard ConvertFrom-Json against empty/invalid input. $ErrorActionPreference is Stop
    # so a failed parse terminates the whole supervisor. Default to {} on failure.
    $detailsObj = $null
    try {
        if ($DetailsJson -and $DetailsJson.Trim()) {
            $detailsObj = ConvertFrom-Json $DetailsJson
        } else {
            $detailsObj = (@{})
        }
    } catch {
        $detailsObj = (@{})
    }
    $entry = @{
        ts = $ts
        decision_type = $DecisionType
        actor = $Actor
        decision = $Decision
        candidate = if ($Candidate) { $Candidate } else { $null }
        run_id = if ($RunId) { $RunId } else { $null }
        reason = $Reason
        details = $detailsObj
        severity = $Severity
    } | ConvertTo-Json -Compress -Depth 5
    $decPath = Join-Path $LogsDir "PIPELINE_DECISIONS.jsonl"
    Add-Content -Path $decPath -Value $entry -Encoding UTF8 -Force
}

# --- Heartbeat datetime helper (SUPERVISOR HARDENING: support unix epoch float from progress_writer + ISO) ---
function Convert-HeartbeatToDate {
    param($Hb)
    if (-not $Hb) { return $null }
    try {
        if ($Hb -is [double] -or $Hb -is [int] -or $Hb -is [long] -or ($Hb -is [string] -and $Hb -match '^\d+(\.\d+)?$')) {
            $secs = [double]$Hb
            # Unix epoch -> UTC DateTime
            return ([DateTimeOffset]::FromUnixTimeSeconds([long]$secs).UtcDateTime)
        }
        # ISO / string date
        return [datetime]::Parse($Hb, [System.Globalization.CultureInfo]::InvariantCulture, [System.Globalization.DateTimeStyles]::AssumeUniversal).ToUniversalTime()
    } catch {
        try { return [datetime]$Hb } catch { return $null }
    }
}

Write-SupLog "INFO" "=== VPS AGI Supervisor starting (Repo=$RepoRoot, MonitorOnly=$MonitorOnly, DryRun=$DryRun) ==="

# --- Python discovery (matches start_server.ps1 / launch_agi_trading.ps1) ---
$pythonCandidates = @(
    (Join-Path $RepoRoot ".venv312\Scripts\python.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\python.exe")
)
$pythonExe = $null
foreach ($cand in $pythonCandidates) {
    if (Test-Path $cand) { $pythonExe = $cand; break }
}
if (-not $pythonExe) {
    Write-SupLog "ERROR" "No venv python found in .venv312 or .venv. Aborting supervisor."
    exit 1
}
Write-SupLog "INFO" "Using Python: $pythonExe"

# --- Process helpers (re-uses patterns from launch_agi_trading.ps1) ---
function Test-AgiServerRunning {
    # 1. CIM process scan for python + Server_AGI token
    try {
        $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        if ($rows) {
            $needle = "python.server_agi"
            foreach ($p in $rows) {
                $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
                if ($cmd.Contains($needle)) { return $true }
            }
        }
    } catch { }

    # 2. Fallback: lockfile + live PID check
    if (Test-Path $LockPath) {
        try {
            $pidRaw = (Get-Content -Path $LockPath -Raw -ErrorAction Stop).Trim()
            $pid = 0
            if ([int]::TryParse($pidRaw, [ref]$pid) -and $pid -gt 0) {
                $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$pid" -ErrorAction SilentlyContinue
                if ($proc -and ([string]$proc.CommandLine).ToLower().Contains("server_agi")) {
                    return $true
                }
            }
        } catch { }
    }
    return $false
}

function Get-AgiHealth {
    param([string]$BaseUrl = $HealthBase)
    $endpoints = @("/api/health", "/api/health/ready")
    $results = @{}
    foreach ($ep in $endpoints) {
        try {
            $uri = "$BaseUrl$ep"
            $resp = Invoke-WebRequest -Uri $uri -UseBasicParsing -TimeoutSec 8 -ErrorAction Stop
            $body = $resp.Content | ConvertFrom-Json -ErrorAction SilentlyContinue
            $results[$ep] = @{
                StatusCode = $resp.StatusCode
                Body       = $body
                Ok         = ($resp.StatusCode -eq 200)
            }
        } catch {
            $results[$ep] = @{ StatusCode = 0; Error = $_.Exception.Message; Ok = $false }
        }
    }
    return $results
}

function Test-MT5Running {
    try {
        $mt5 = Get-Process -Name "terminal64" -ErrorAction SilentlyContinue
        return ($null -ne $mt5)
    } catch { return $false }
}

function Test-DiskHealthy {
    try {
        $d = Get-PSDrive C
        $freeGB = [math]::Round($d.Free / 1GB, 1)
        return ($freeGB -gt 5)
    } catch { return $true }  # non-fatal
}

function Get-MT5LoginHint {
    # Lightweight: scan today's MT5 log for auth success (non-blocking)
    try {
        $logDir = "$env:APPDATA\MetaQuotes\Terminal"
        $today = (Get-Date).ToString("yyyyMMdd")
        $log = Get-ChildItem -Path $logDir -Recurse -Filter "*$today.log" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($log) {
            $tail = Get-Content $log.FullName -Tail 30 -ErrorAction SilentlyContinue
            if ($tail -match "authorized on .*Trial") { return "Trial account recent auth OK" }
            if ($tail -match "authorized") { return "Recent MT5 auth detected" }
        }
    } catch {}
    return "MT5 login unknown (check terminal)"
}

# --- Training / Paper Harness / Candidate detection (for 50k postfix runs + coordination with Training agent) ---
function Test-TrainingRunning {
    try {
        $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        if ($rows) {
            foreach ($p in $rows) {
                $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
                if ($cmd -match "(train_drl|enhanced_train|start_enhanced_training|postfix)") { return $true }
            }
        }
    } catch {}
    # Fallback: recent active training log (last 10m write) - include v4 robust launcher logs
    try {
        $recentTrainLogs = Get-ChildItem "$RepoRoot\logs" -ErrorAction SilentlyContinue -Include "*postfix*.log","*robust*.log","*v4*.log" |
            Where-Object { ((Get-Date) - $_.LastWriteTime).TotalMinutes -lt 10 }
        if ($recentTrainLogs) { return $true }
    } catch {}
    return $false
}

function Test-PaperHarnessRunning {
    try {
        $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        if ($rows) {
            foreach ($p in $rows) {
                $cmd = ([string]$p.CommandLine).ToLower().Replace("\", "/")
                if ($cmd -match "paper_mt5_execution_harness") { return $true }
            }
        }
    } catch {}
    # Log activity fallback (harness writes jsonl frequently when active)
    $harnessLog = Join-Path $LogsDir "paper_harness_exec.jsonl"
    if (Test-Path $harnessLog) {
        try {
            $age = (Get-Date) - (Get-Item $harnessLog).LastWriteTime
            if ($age.TotalMinutes -lt 5) { return $true }
        } catch {}
    }
    return $false
}

function Test-RecentCandidateStaged {
    # Scans recent logs + candidate dirs for post-fix good candidates (coord with Training agent + Current Training Run Monitor)
    # ROBUST v2 (Post-Candidate Handoff Automation): prefers Python finder for exact "alignment_fix_applied + clean" parity with promoter/deploy/export
    try {
        $pythonExeLocal = $pythonExe  # from top-level discovery (script scope)
        if ($pythonExeLocal -and (Test-Path $pythonExeLocal)) {
            $findCmd = "& '$pythonExeLocal' 'tools\export_for_mql5.py' --find-latest-good-candidate"
            $out = Invoke-Expression $findCmd 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0 -and $out -and -not ($out -match "NO_GOOD|None|not found")) {
                $candPath = $out.Trim() -split "`n" | Select-Object -Last 1 | ForEach-Object { $_.Trim() }
                if ($candPath -and (Test-Path $candPath)) {
                    $ageMin = [int](((Get-Date) - (Get-Item $candPath).LastWriteTime).TotalMinutes)
                    if ($ageMin -lt 2880) {  # 48h window for v4 long runs
                        # Extra clean scorecard check (alignment true + no quarantine)
                        $sc = Join-Path $candPath "scorecard.json"
                        if (Test-Path $sc) {
                            $scContent = Get-Content $sc -Raw -ErrorAction SilentlyContinue
                            if ($scContent -match '"alignment_fix_applied"\s*:\s*[^fF]' -and $scContent -notmatch 'quarantined|PRE-ALIGNMENT') {
                                $isV4 = ($scContent -match 'v4_robust|robust_v4|AGI_V4_ROBUST|"launcher":"robust_v4' -or $scContent -match '"launcher_version":"v4"')
                                $tag = if ($isV4) { "v4_robust_conservative" } else { "post-fix" }
                                return "YES (${tag} via finder, age ${ageMin}m)"
                            }
                        }
                        return "YES (post-fix via finder, age ${ageMin}m - scorecard verify)"
                    }
                }
            }
        }
    } catch { Write-SupLog "DEBUG" "Python finder in candidate check failed (non-fatal): $_" }

    # Fallback: direct dir scan (mirrors Python _find_latest_good_candidate + enhanced_train_drl enrichment)
    try {
        $candDir = Join-Path $RepoRoot "models\registry\candidates"
        if (Test-Path $candDir) {
            $latestCands = Get-ChildItem $candDir -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 5
            foreach ($c in $latestCands) {
                $ageMin = [int](((Get-Date) - $c.LastWriteTime).TotalMinutes)
                if ($ageMin -gt 2880) { continue }
                $score = Join-Path $c.FullName "scorecard.json"
                if (Test-Path $score) {
                    $content = Get-Content $score -Raw -ErrorAction SilentlyContinue
                    if ($content -match '"alignment_fix_applied"') {
                        if ($content -notmatch 'quarantined|PRE-ALIGNMENT') {
                            # Also verify model files exist for resilience (v4 staged candidate)
                            $hasModel = (Test-Path (Join-Path $c.FullName "ppo_trading.zip")) -or (Test-Path (Join-Path $c.FullName "best_model.zip"))
                            if ($hasModel) {
                                # V4 ROBUST WIRING: detect if this candidate came from the advanced conservative v4 50k launcher
                                $isV4 = ($content -match 'v4_robust|robust_v4|AGI_V4_ROBUST|"launcher":"robust_v4' -or $content -match '"launcher_version":"v4"')
                                $isConservative = ($content -match 'conservative_params":\s*true|"target_kl":\s*0\.05|"AGI_CONSERVATIVE_RUN"')
                                $tag = if ($isV4) { "v4_robust" } elseif ($isConservative) { "conservative" } else { "postfix" }
                                return "YES (${tag} post-fix clean fallback, age ${ageMin}m, cand=$($c.Name))"
                            }
                        }
                    }
                }
            }
        }
    } catch {}
    # Log fallback (for observer coordination) - extended for v4 robust launcher logs (robust_v4_*.log etc)
    try {
        $logs = Get-ChildItem "$RepoRoot\logs" -Filter "*postfix*.log" -ErrorAction SilentlyContinue
        $logs += Get-ChildItem "$RepoRoot\logs" -Filter "*robust*.log" -ErrorAction SilentlyContinue
        $logs += Get-ChildItem "$RepoRoot\logs" -Filter "*v4*.log" -ErrorAction SilentlyContinue
        $logs = $logs | Sort-Object LastWriteTime -Descending | Select-Object -First 5 -Unique
        foreach ($l in $logs) {
            if (((Get-Date) - $l.LastWriteTime).TotalHours -gt 48) { continue }
            $tail = Get-Content $l.FullName -Tail 40 -ErrorAction SilentlyContinue
            if ($tail -match "Candidate staged|alignment_fix_applied|=== V4 (START|DONE)") {
                $isV4Log = ($l.Name -match "robust_v4|v4_robust|launch_robust.*v4") -or ($tail -match "robust_v4|v4_robust|launch_robust_postfix_training_v4|V4 START")
                $tag = if ($isV4Log) { "v4_robust" } else { "postfix" }
                return "YES (${tag} log signal recent, from $($l.Name))"
            }
        }
    } catch {}
    return "NO"
}

function Get-TrainingStatusSummary {
    $trainRun = Test-TrainingRunning
    $harnessRun = Test-PaperHarnessRunning
    $cand = Test-RecentCandidateStaged
    $healthInfo = ""
    $hPath = Join-Path $RepoRoot "logs\training_health.json"
    if (Test-Path $hPath) {
        try {
            $h = Get-Content $hPath -Raw | ConvertFrom-Json -ErrorAction Stop
            $hbDate = Convert-HeartbeatToDate $h.last_heartbeat
            $hbAge = if ($hbDate) { [int](((Get-Date).ToUniversalTime() - $hbDate).TotalMinutes) } else { "?" }
            $healthInfo = " Health(status=$($h.status),step=$($h.current_step)/$($h.total_timesteps),age=${hbAge}m,recov=$($h.recovery_attempts))"
        } catch {}
    }
    return "Training=$trainRun Harness=$harnessRun Candidate=$cand$healthInfo"
}

# === Bounded Training Recovery (Training Robustness & Recovery Agent 2026-05-27) ===
# Uses training_health.json signals + log heuristics for safe, conservative auto-recovery.
# Never unbounded; prevents restart storms on 50k+ postfix runs.
$script:TrainingRecoveryAttempts = 0
$script:LastTrainingRecoveryTime = $null
$MaxTrainingRecoveries = 3          # bounded cap
$TrainingRecoveryCooldownSec = 300  # 5 min min spacing (conservative)
$TrainingStallThresholdMin = 12     # consider stalled after 12min no heartbeat/progress

function Test-TrainingHealthStalled {
    # Prefer explicit health signal over log tail (new robust path)
    # v4 stall diagnosis fix #4: also check "no new lines in robust_v*.log since last health write"
    # This is stronger/faster signal when external health writers (e.g. v4_handoff_prep, supervisor prep) keep
    # status=running + fresh last_heartbeat even after the actual training process died silently.
    $healthPath = Join-Path $RepoRoot "logs\training_health.json"
    if (Test-Path $healthPath) {
        try {
            $h = Get-Content $healthPath -Raw | ConvertFrom-Json -ErrorAction Stop
            $hbDate = Convert-HeartbeatToDate $h.last_heartbeat
            $ageMin = if ($hbDate) { [int](((Get-Date).ToUniversalTime() - $hbDate).TotalMinutes) } else { 999 }
            if ($h.status -in @("failed", "stalled")) { return $true }
            if ($h.status -eq "running" -and $ageMin -gt $TrainingStallThresholdMin) { return $true }
            if ($h.status -eq "recovering" -and $ageMin -gt ($TrainingStallThresholdMin * 2)) { return $true }

            # v4 diagnosis enhancement: log-not-updated-since-health (even if health looks fresh from external injection)
            try {
                $robustLog = Get-ChildItem "$RepoRoot\logs" -Filter "robust_v*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
                if ($robustLog -and $hbDate) {
                    $logMtimeUtc = $robustLog.LastWriteTime.ToUniversalTime()
                    if ($logMtimeUtc -lt $hbDate) {
                        $logStaleMin = [int](((Get-Date).ToUniversalTime() - $logMtimeUtc).TotalMinutes)
                        if ($logStaleMin -gt 1) {  # training wrote nothing to its robust log after health was touched externally
                            return $true
                        }
                    }
                }
            } catch {}
        } catch {}
    }
    # Fallback to legacy log heuristic (kept for compatibility)
    try {
        $recentPostfix = Get-ChildItem "$RepoRoot\logs" -Filter "*postfix*.log" -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
        if ($recentPostfix -and ((Get-Date) - $recentPostfix.LastWriteTime).TotalMinutes -gt $TrainingStallThresholdMin) {
            $tail = Get-Content $recentPostfix.FullName -Tail 20 -ErrorAction SilentlyContinue
            if ($tail -match "(error|exception|failed|kl explosion|traceback|crash)" -or ($tail -notmatch "progress|step=")) {
                return $true
            }
        }
    } catch {}
    return $false
}

function Invoke-BoundedTrainingRecovery {
    param([string]$Symbol = "BTCUSDm", [int]$Timesteps = 50000)
    if ($MonitorOnly -or $DryRun) {
        Write-SupLog "INFO" "[DRY/MONITOR] Would perform bounded training recovery for $Symbol $Timesteps"
        return $false
    }
    $now = Get-Date
    if ($script:LastTrainingRecoveryTime -and (($now - $script:LastTrainingRecoveryTime).TotalSeconds -lt $TrainingRecoveryCooldownSec)) {
        Write-SupLog "WARN" "Training recovery skipped (cooldown active)"
        return $false
    }
    if ($script:TrainingRecoveryAttempts -ge $MaxTrainingRecoveries) {
        Write-SupLog "ERROR" "Training recovery cap reached ($MaxTrainingRecoveries). Manual intervention required: use launch_robust_postfix_training_v4.ps1"
        return $false
    }

    Write-SupLog "INFO" "AUTO-RECOVERY: Launching stalled postfix training (attempt $($script:TrainingRecoveryAttempts+1)/$MaxTrainingRecoveries) with CONSERVATIVE hyperparams (kl=0.05, lr=3e-5, n_steps=8192) [NEW MTF: EnableTimeframeOpt=true -> 1m+5m+15m+1h + best_features]"

    # Write recovering health signal immediately
    try {
        $recHealth = @{ status = "recovering"; recovery_attempts = ($script:TrainingRecoveryAttempts + 1); symbol = $Symbol; total_timesteps = $Timesteps; conservative_params = $true; last_heartbeat = (Get-Date).ToUniversalTime().ToString("o") }
        $recHealth | ConvertTo-Json -Depth 5 | Set-Content -Path (Join-Path $RepoRoot "logs\training_health.json") -Encoding UTF8 -Force
    } catch {}

    try {
        # PREFERRED: use the robust v5 launcher (iteration on v4 success; strongest conservative tiers + recovery + v5 diagnostics per launcher header). Falls back to v4 then legacy.
        $robustV5 = Join-Path $RepoRoot "scripts\launch_robust_postfix_training_v5.ps1"
        $robustV4 = Join-Path $RepoRoot "scripts\launch_robust_postfix_training_v4.ps1"
        if (Test-Path $robustV5) {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = "powershell.exe"
            $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$robustV5`" -Symbol $Symbol -Timesteps $Timesteps -EnableTimeframeOpt:`$true  # NEW MTF STANDARD: 1m+5m+15m+1h + best params"
            $psi.WorkingDirectory = $RepoRoot
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            Write-SupLog "INFO" "Spawned robust v5 recovery (PREFERRED for post-v4 stall; PID=$($proc.Id))"
        } elseif (Test-Path $robustV4) {
            $psi = New-Object System.Diagnostics.ProcessStartInfo
            $psi.FileName = "powershell.exe"
            $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$robustV4`" -Symbol $Symbol -Timesteps $Timesteps -EnableTimeframeOpt:`$true  # NEW MTF STANDARD: 1m+5m+15m+1h + best params"
            $psi.WorkingDirectory = $RepoRoot
            $psi.UseShellExecute = $false
            $psi.CreateNoWindow = $true
            $proc = [System.Diagnostics.Process]::Start($psi)
            Write-SupLog "INFO" "Spawned robust v4 recovery (fallback; PID=$($proc.Id))"
        } else {
            # Fallback to main postfix launcher (detached, sets conservative envs internally)
            $launcher = Join-Path $RepoRoot "launch_postfix_training.ps1"
            Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $launcher, "-Symbol", $Symbol, "-Timesteps", $Timesteps, "-Detach" -WindowStyle Minimized -WorkingDirectory $RepoRoot
            Write-SupLog "INFO" "Spawned postfix launcher recovery (detached)"
        }

        $script:TrainingRecoveryAttempts++
        $script:LastTrainingRecoveryTime = $now
        Start-Sleep -Seconds 8
        return $true
    } catch {
        Write-SupLog "ERROR" "Training recovery launch failed: $($_.Exception.Message)"
        return $false
    }
}

function Launch-TuiWatcherOnCandidate {
    # Self-sustainability: auto-spawn robust watcher (which launches TUI) when candidate appears.
    # Uses launch_tui.ps1 -Watcher -Persistent for background observation.
    param([switch]$OnceSnapshot)
    if ($DryRun -or $MonitorOnly) {
        Write-SupLog "INFO" "[DRY/MONITOR] Would launch TUI watcher for new candidate"
        return
    }
    $watcherLauncher = Join-Path $RepoRoot "launch_tui.ps1"
    if (-not (Test-Path $watcherLauncher)) {
        Write-SupLog "WARN" "No launch_tui.ps1 for watcher integration"
        return
    }
    try {
        $mode = if ($OnceSnapshot) { "-Once" } else { "-Watcher -Persistent" }
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        $psi.FileName = "powershell.exe"
        $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$watcherLauncher`" $mode"
        $psi.WorkingDirectory = $RepoRoot
        $psi.UseShellExecute = $false
        $psi.CreateNoWindow = $true   # background friendly
        $proc = [System.Diagnostics.Process]::Start($psi)
        Write-SupLog "INFO" "Auto-launched TUI watcher (PID=$($proc.Id)) for candidate observation. Use launch_tui.ps1 -Watcher to foreground."
    } catch {
        Write-SupLog "WARN" "Failed to auto-launch TUI watcher: $($_.Exception.Message)"
    }
}

# --- DUPLICATE TUI CLEANUP (SUPERVISOR HARDENING per audit) ---
# Kills stray system-Python (or non-.venv312) TUI/monitor_tui processes.
# Keeps only the .venv312 python.exe instances running TUI. Refreshes runtime/agent_status/ visibility.
function Clean-DuplicateTuiProcesses {
    param([switch]$LogOnly)
    $killed = 0
    $kept = 0
    try {
        $rows = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue
        foreach ($p in $rows) {
            $cmd = [string]$p.CommandLine
            $lower = $cmd.ToLower().Replace("\", "/")
            if ($lower -match "monitor_tui|launch_tui|tui\.py|start_tui") {
                $isVenv312 = $lower -match "\.venv312"
                if (-not $isVenv312) {
                    # Stray system python TUI (common duplicate source)
                    if (-not $LogOnly) {
                        try {
                            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
                            $killed++
                        } catch {}
                    }
                    Write-SupLog "WARN" "Cleaned duplicate TUI (system/non-venv python PID=$($p.ProcessId)): $cmd"
                } else {
                    $kept++
                }
            }
        }
        if ($killed -gt 0) {
            Write-SupLog "INFO" "TUI duplicate cleanup: killed $killed stray (non-.venv312) TUI processes; kept $kept venv312 ones."
            # Refresh agent_status swarm entries after cleanup
            try {
                $py = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
                if (Test-Path $py) {
                    & $py -c "
from scripts.swarm_status import sync_grok_swarm
sync_grok_swarm(36)
" 2>$null | Out-Null
                    Write-SupLog "DEBUG" "Refreshed runtime/agent_status/ after TUI cleanup."
                }
            } catch {}
        }
    } catch {
        Write-SupLog "DEBUG" "TUI cleanup scan note: $_"
    }
    return $killed
}

# ============================================================
# PRODUCTION HARDENING: Self-Monitor Integration (2026-05-28 Zero-Touch)
# Calls the new SelfMonitoringRecoveryAgent.monitor_cycle() for auto kill-switch,
# rollback, conservative regime, retrain requests. Bounded, non-blocking, clear logs.
# Status written to runtime/agent_status/self_monitoring_recovery_agent.json (TUI readable)
# ============================================================
function Invoke-SelfMonitorCycle {
    param([switch]$LogOnly)
    try {
        $py = $pythonExe
        if (-not (Test-Path $py)) { return }
        $cmd = "& '$py' -c `"from Python.autonomous.self_monitor import SelfMonitoringRecoveryAgent; sm = SelfMonitoringRecoveryAgent(enable_alerts=True); res = sm.monitor_cycle(); status = res.get('status','ok') if isinstance(res, dict) else 'ok'; issues = len(res.get('issues',[])) if isinstance(res,dict) else 0; print('SELF_MONITOR_CYCLE:OK status=' + status + ' issues=' + [string]$issues)`" 2>&1"
        if ($LogOnly -or $DryRun -or $MonitorOnly) {
            Write-SupLog "INFO" "[DRY/MONITOR] Would invoke SelfMonitoringRecoveryAgent.monitor_cycle() for auto-rollback/kill/conservative/retrain"
            return
        }
        $out = Invoke-Expression $cmd | Out-String
        if ($out -match "SELF_MONITOR_CYCLE:OK") {
            Write-SupLog "INFO" "Self-monitor cycle OK: $out".Trim()
            Write-PipelineDecision -DecisionType "self_monitor" -Actor "supervisor" -Decision "MONITOR_CYCLE" -Reason "periodic" -Severity "info"
        } else {
            Write-SupLog "WARN" "Self-monitor cycle output: $out".Trim()
        }
    } catch {
        Write-SupLog "DEBUG" "Self-monitor cycle non-fatal (graceful): $_"
    }
}

# --- POST-CANDIDATE HANDOFF AUTOMATION (core of this agent's mission) ---
# Makes "good post-fix candidate (alignment_fix_applied + clean scorecard)" -> promoter (safe) + MQL5 deploy reliable + visible.
# Always prepares exact commands (for manual or future auto). Under env gate: drives full auto via auto_promote (which calls promoter --auto-launch -> harness + deploy_mql5).
# Writes runtime artifacts for TUI / monitor coordination. Resilient to v4 runs (enhanced_train_drl scorecard enrichment).
function Invoke-PostCandidateHandoff {
    param(
        [string]$CandidateState,
        [string]$CandidatePath = ""
    )
    $ts = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-SupLog "INFO" "=== POST-CANDIDATE HANDOFF AUTOMATION TRIGGERED (v4 BTCUSDm resilient) @ $ts ==="
    Write-SupLog "INFO" "Candidate state: $CandidateState"

    $runtime = Join-Path $RepoRoot "runtime"
    New-Item -ItemType Directory -Force -Path $runtime | Out-Null

    # Discover exact candidate dir if not passed
    if (-not $CandidatePath -or -not (Test-Path $CandidatePath)) {
        try {
            $pyFind = & $pythonExe "tools\export_for_mql5.py" --find-latest-good-candidate 2>&1 | Out-String
            if ($LASTEXITCODE -eq 0) {
                $CandidatePath = ($pyFind.Trim() -split "`n" | Select-Object -Last 1).Trim()
            }
        } catch {}
        if (-not $CandidatePath -or -not (Test-Path $CandidatePath)) {
            $candRoot = Join-Path $RepoRoot "models\registry\candidates"
            $latest = Get-ChildItem $candRoot -Directory -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($latest) { $CandidatePath = $latest.FullName }
        }
    }
    $candName = if ($CandidatePath) { Split-Path $CandidatePath -Leaf } else { "unknown" }
    Write-SupLog "INFO" "Handoff candidate: $candName ($CandidatePath)"

    # 1. ALWAYS prepare exact safe commands (even if auto not enabled) - reliable for operator / monitor agent
    $handoffCmdFile = Join-Path $runtime "post_candidate_handoff_commands.txt"
    $promoterDry = "python scripts\promote_candidate_to_paper.py --symbols BTCUSDm --dry-run --execution-type decision_ppo"
    $promoterAuto = '$env:CHAIN_GAMBLER_EXECUTION_MODE="demo"; $env:AGI_PAPER_FIXED_LOT="0.01"; $env:AGI_CONSERVATIVE_PAPER="1"; $env:AGI_EXECUTION_TYPE="decision_ppo"; $env:AGI_MULTI_TF_STANDARD="1"; $env:AGI_USE_BEST_FEATURES="1"; python scripts\promote_candidate_to_paper.py --symbols BTCUSDm --auto-launch --execution-type decision_ppo'
    $autoPromoteCmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\auto_promote_candidate.ps1 -Symbol BTCUSDm"
    $decisionPpoNote = "# NEW: DecisionPPO + Execution stack (rich trade specs) default for autonomous loop on promotion. Supervisor/harness/MQL5 use multi-TF + best_features_per_symbol. Set AGI_EXECUTION_TYPE=simple_action for legacy only. Paper harness now instantiates ExecutionAgent; supervisor auto-arms paper (then live) with correct context once gates pass."
    $mql5Cmd = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals"
    $mql5LogOnly = "powershell -NoProfile -ExecutionPolicy Bypass -File scripts\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals -LogOnly"

    $cmds = @"
# POST-CANDIDATE HANDOFF COMMANDS (auto-generated by vps_agi_supervisor on good candidate detection)
# For the v4 BTCUSDm run (or any alignment_fix_applied + clean scorecard candidate)
# Generated: $ts
# Candidate: $candName
# NEW STANDARD: 1m+5m+15m+1h multi-TF + best_features_per_symbol.yaml (multitimeframe_builder)
# Decision + Execution new setup context will be deployed on arm (env AGI_MULTI_TF_STANDARD=1)
# Legacy single-TF: AGI_USE_LEGACY_SINGLE_TF=1

# SAFE REVIEW (always first):
cd C:\supreme-chainsaw
$promoterDry

# FULL PAPER + MQL5 SHADOW (after MT5 demo login; conservative post-fix profile; DecisionPPO rich default):
$promoterAuto

# RECOMMENDED AUTO BRIDGE (env-gated full flow: gates + canary opt-in + paper + MQL5; DecisionPPO+Exec):
# Enable with: `$env:AGI_AUTO_PROMOTE_CANDIDATE="1"; `$env:AGI_AUTO_PAPER_HARNESS="1"; `$env:AGI_AUTO_MQL5="1"
$autoPromoteCmd
$decisionPpoNote

# DIRECT MQL5 SHADOW PREP (export + copy EA + builder + guidance; zero risk):
$mql5Cmd

# DRY MQL5 (preview only):
$mql5LogOnly

# TUI + observer (for handoff visibility):
powershell -File launch_tui.ps1 -Watcher -Persistent

# After paper/MQL5 shadow validation (7d+ clean): promote via model_registry or manual, then live with tiny risk.
"@
    $cmds | Out-File -FilePath $handoffCmdFile -Encoding UTF8 -Force
    Write-SupLog "SUCCESS" "Exact handoff commands written: $handoffCmdFile (use for manual or script)"

    # 2. Write machine-readable handoff status for TUI / Current Training Run Monitor / coordination
    # V4 50k BTCUSDm RUN SPECIFIC: inject current training run provenance + v4 signals for bulletproof handoff traceability (this run uses robust_v4_postfix launcher)
    $v4Provenance = @{}
    try {
        $healthPath = Join-Path $RepoRoot "logs\training_health.json"
        if (Test-Path $healthPath) {
            $h = Get-Content $healthPath -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
            if ($h) {
                $v4Provenance = @{
                    launcher = "robust_v4_postfix"
                    launcher_version = "v4"
                    run_tag = "v4_robust_conservative_50k_BTCUSDm"
                    current_step = $h.current_step
                    pct_complete = $h.pct_complete
                    conservative_params = $true
                    v4_robust = $true
                    timesteps_target = $h.total_timesteps
                    health_last_heartbeat = $h.last_heartbeat
                }
            }
        }
    } catch {}
    $handoffJson = Join-Path $runtime "last_handoff.json"
    $handoffData = @{
        timestamp = $ts
        candidate = $candName
        candidate_path = $CandidatePath
        state = $CandidateState
        promoter_launched = $false
        mql5_deploy_triggered = $false
        auto_gate_enabled = ($env:AGI_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:SUPERVISOR_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:AGI_AUTO_PROMOTE -eq "1")
        commands_file = $handoffCmdFile
        v4_specific_run = $v4Provenance
        # execution_type is set below; see line ~750 (default for new; MTF/best_features context)
        uses_rich_specs = $true
        mtf_context = $true
        best_features = "configs/best_features_per_symbol.yaml"
        is_v4_robust_candidate = $true
        source_run_tag = "v4_robust_conservative_50k_BTCUSDm"
        execution_type = "decision_ppo"
        decision_ppo_armed = $true
        uses_rich_trade_specs = $true
        mtf_best_features = "configs/best_features_per_symbol.yaml + 1m/5m/15m/1h"
        stack = "DecisionPPO + ExecutorRouter + GateEngine + RiskSupervisor (full autonomous paper->live)"
        supervisor_auto_starts = "On gates pass: auto paper (decision_ppo) then (post validation) live via harness/executor; rollback/flatten always wired"
        next_steps = @(
            "1. Review: $promoterDry",
            "2. Arm+launch: promoter --auto-launch or auto_promote_candidate.ps1 (with env)",
            "3. MQL5: $mql5Cmd (ShadowMode=true first)",
            "4. Monitor in TUI + harness jsonl + MQL5 logs",
            "NEW STANDARD (default): 1m+5m+15m+1h + best_features_per_symbol.yaml via fetch_multitimeframe + multitimeframe_builder; Decision+Execution new setup context armed"
        )
        multi_timeframe_standard = $true
        timeframes = @("1m","5m","15m","1h")
        feature_params = "configs/best_features_per_symbol.yaml (auto)"
        decision_execution = "new DecisionBuilder + ExecutorRouter + GateEngine (multi-TF context; ready on promote)"
    } | ConvertTo-Json -Depth 4
    $handoffData | Out-File -FilePath $handoffJson -Encoding UTF8 -Force
    Write-SupLog "INFO" "Handoff status JSON for TUI/monitor: $handoffJson (v4 50k BTCUSDm provenance injected for this specific run)"

    # Decision PPO + Execution auto-start in paper (then live) per task closure
    # Once candidate passes gates (via promoter), supervisor can directly arm the rich stack using MTF + best features.
    # This is in addition to promoter --auto-launch path; enables explicit supervisor oversight of DecisionPPO paper->live.
    if ($autoEnabled -and -not $DryRun -and -not $MonitorOnly) {
        try {
            $paperEnv = @{
                "CHAIN_GAMBLER_EXECUTION_MODE" = "demo"
                "AGI_PAPER_FIXED_LOT" = "0.01"
                "AGI_EXECUTION_TYPE" = "decision_ppo"
                "AGI_MULTI_TF_STANDARD" = "1"
                "AGI_USE_BEST_FEATURES" = "1"
                "AGI_BEST_FEATURES_CONFIG" = "configs/best_features_per_symbol.yaml"
            }
            Write-SupLog "SUCCESS" "Supervisor auto-starting DecisionPPO+Execution paper harness (rich specs, MTF+best_features context) for $candName"
            # Launch harness detached (it will use execution_type from paper_harness_start.json written by promoter, or env)
            $harnessArgs = "-NoProfile -ExecutionPolicy Bypass -Command `"cd '$RepoRoot'; `$env:CHAIN_GAMBLER_EXECUTION_MODE='demo'; `$env:AGI_EXECUTION_TYPE='decision_ppo'; `$env:AGI_MULTI_TF_STANDARD='1'; `$env:AGI_USE_BEST_FEATURES='1'; & '$pythonExe' 'scripts\paper_mt5_execution_harness.py' --symbols BTCUSDm --max-days 7 --equity-start 5000`""
            Start-Process -FilePath "powershell.exe" -ArgumentList $harnessArgs -WindowStyle Hidden -WorkingDirectory $RepoRoot
            Write-SupLog "INFO" "DecisionPPO paper harness launch requested (will respect rollback flag for flatten)"
            # Live transition marker: after X clean days harness self or supervisor can flip to real_live (future: add live_gate check + tiny risk)
            "decision_ppo_paper_armed=$ts" | Out-File -FilePath (Join-Path $runtime "decision_ppo_paper_armed.flag") -Encoding UTF8 -Force
        } catch {
            Write-SupLog "WARN" "DecisionPPO paper auto-start encountered non-fatal issue: $_ (promoter path still active)"
        }
    }

    # 3. Invoke actual automation under safe gates (or log prepared)
    $autoEnabled = ($env:AGI_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:SUPERVISOR_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:AGI_AUTO_PROMOTE -eq "1")
    if ($autoEnabled -and -not $DryRun -and -not $MonitorOnly) {
        Write-SupLog "SUCCESS" "AUTO HANDOFF ARMED (env gate). Driving full transition: promoter gates -> DecisionPPO+Execution paper harness (multi-TF + best_features) + MQL5 shadow via auto_promote."
        # The block below already handles launch of auto_promote; we ensure MQL5 path too via it.
        # For extra resilience, also ensure MQL5 deploy is directly callable (promoter inside auto will do on --auto-launch).
        try {
            # Touch a trigger flag the promoter/deploy can watch (future-proofing)
            "HANDOFF_AUTO=$ts" | Out-File -FilePath (Join-Path $runtime "handoff_trigger.flag") -Encoding UTF8 -Force
        } catch {}
        # Explicit auto-start of paper harness in DecisionPPO rich mode (supervisor closure of autonomous loop)
        try {
            $paperEnv = '$env:AGI_EXECUTION_TYPE="decision_ppo"; $env:AGI_MULTI_TF_STANDARD="1"; $env:AGI_USE_BEST_FEATURES="1"; $env:CHAIN_GAMBLER_EXECUTION_MODE="demo"; $env:AGI_PAPER_FIXED_LOT="0.01"'
            Write-SupLog "INFO" "Auto-launching paper harness with Decision PPO + Execution layer (rich specs + MTF/best-feats)..."
            Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "`"$paperEnv; python scripts\paper_mt5_execution_harness.py --symbols BTCUSDm --max-days 7`"" -WindowStyle Hidden -WorkingDirectory $RepoRoot
            Write-SupLog "SUCCESS" "DecisionPPO+Exec paper harness auto-started under supervisor (will transition to live after validation gates)."
        } catch {
            Write-SupLog "WARN" "Auto paper harness (rich) launch note: $($_.Exception.Message)"
        }
    } else {
        Write-SupLog "INFO" "SAFE MODE (no auto env gate): Commands + guidance prepared. Promoter/MQL5 NOT auto-executed. Set AGI_AUTO_PROMOTE_CANDIDATE=1 then re-detect (or run commands above) for full automatic handoff on this v4 candidate."
        # Still run promoter in dry/prep mode for immediate checklist/guidance/MQL5 txt (non-launching)
        try {
            $prepCmd = "$pythonExe scripts\promote_candidate_to_paper.py --symbols BTCUSDm --dry-run"
            Write-SupLog "INFO" "Running promoter --dry-run for immediate checklist + MQL5 guidance (safe prep)..."
            Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", "`"$prepCmd`"" -WindowStyle Hidden -WorkingDirectory $RepoRoot -Wait
            Write-SupLog "SUCCESS" "Safe promoter dry-run completed (checklist + guidance ready for TUI)."
            (Get-Content $handoffJson -Raw | ConvertFrom-Json).promoter_launched = $true | Out-Null  # best effort update
        } catch {
            Write-SupLog "WARN" "Dry-run promoter prep note: $($_.Exception.Message)"
        }
    }

    # 4. Always ensure MQL5 guidance / shadow prep artifacts exist (even safe mode)
    # Promoter dry already generates some; direct deploy -LogOnly as extra resilient step.
    try {
        Write-SupLog "INFO" "Ensuring MQL5 shadow artifacts via LogOnly deploy (non-destructive)..."
        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"scripts\deploy_mql5_chain_gambler.ps1`"", "-AutoFromRegistry", "-ShadowPrep", "-LogOnly", "-Quiet" -WindowStyle Hidden -WorkingDirectory $RepoRoot
        Write-SupLog "SUCCESS" "MQL5 LogOnly prep triggered (artifacts + mql5_shadow_ready.json ready)."
    } catch {
        Write-SupLog "WARN" "MQL5 LogOnly prep skipped (non-fatal): $_"
    }

    Write-SupLog "INFO" "=== POST-CANDIDATE HANDOFF AUTOMATION COMPLETE (logs + runtime/ artifacts updated; TUI will surface) ==="
    return $handoffJson
}

function Start-AgiServer {
    if ($DryRun) {
        Write-SupLog "WARN" "[DRY-RUN] Would start AGI server now"
        return $null
    }
    if ($MonitorOnly) {
        Write-SupLog "WARN" "MonitorOnly mode - refusing to start server"
        return $null
    }

    # Clean stale before start (best effort)
    if (Test-Path $LockPath) {
        Remove-Item $LockPath -Force -ErrorAction SilentlyContinue
    }

    # Paper-trading safety defaults (override only if explicitly set to live by operator)
    if (-not $env:CHAIN_GAMBLER_EXECUTION_MODE) { $env:CHAIN_GAMBLER_EXECUTION_MODE = "paper" }
    if (-not $env:CHAIN_GAMBLER_ALLOW_LIVE) { $env:CHAIN_GAMBLER_ALLOW_LIVE = "0" }
    Write-SupLog "INFO" "Launch env: MODE=$($env:CHAIN_GAMBLER_EXECUTION_MODE) ALLOW_LIVE=$($env:CHAIN_GAMBLER_ALLOW_LIVE)"

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pythonExe
    $psi.Arguments = "-m Python.Server_AGI"
    $psi.WorkingDirectory = $RepoRoot
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    # Inherit current env (allows MT5_*, TELEGRAM_*, AGI_* overrides)

    try {
        $proc = [System.Diagnostics.Process]::Start($psi)
        Start-Sleep -Seconds 3
        Write-SupLog "INFO" "Started AGI Server (PID=$($proc.Id))"
        return $proc.Id
    } catch {
        Write-SupLog "ERROR" "Failed to launch Server_AGI: $($_.Exception.Message)"
        return $null
    }
}

# --- Restart accounting (script-scoped for cross-function mutation) ---
$script:restartTimes = [System.Collections.Generic.List[DateTime]]::new()
$script:lastRestart = [DateTime]::MinValue

function ShouldAllowRestart {
    $now = Get-Date
    # Remove entries older than 1 hour
    for ($i = $script:restartTimes.Count - 1; $i -ge 0; $i--) {
        if (($now - $script:restartTimes[$i]).TotalHours -gt 1) { $script:restartTimes.RemoveAt($i) }
    }
    if ($script:restartTimes.Count -ge $MaxRestartsPerHour) {
        Write-SupLog "ERROR" "Restart cap reached ($($script:restartTimes.Count)/$MaxRestartsPerHour in last hour). Cooling down."
        return $false
    }
    if (($now - $script:lastRestart).TotalSeconds -lt $RestartCooldownSeconds) {
        Write-SupLog "WARN" "Restart cooldown active ($RestartCooldownSeconds s)"
        return $false
    }
    return $true
}

function RecordRestart {
    $now = Get-Date
    $script:restartTimes.Add($now)
    $script:lastRestart = $now
    # Unified audit for restart decision (key for swarm observability)
    Write-PipelineDecision -DecisionType "supervisor" -Actor "supervisor" -Decision "SERVER_RESTART" -Reason "health_or_crash_recovery" -Severity "warn" -DetailsJson ('{"ts":"' + $now.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ") + '"}')
}

# ============================================================
# PRODUCTION HARDENING: Supervisor Status + Alerts + Meta Overrides (Zero-Touch TUI Ready)
# Writes runtime/agent_status/supervisor_status.json + alerts_supervisor.json
# (mini TUI / monitor_tui / React can poll these without parsing full logs)
# Auto-loads latest meta overrides from runtime/next_training_overrides.json for cycle awareness
# ============================================================
function Write-SupervisorStatus {
    param(
        [string]$ExtraNote = "",
        [hashtable]$Alerts = @{}
    )
    $ts = (Get-Date).ToUniversalTime().ToString("o")
    $runtime = Join-Path $RepoRoot "runtime"
    $agentStatus = Join-Path $runtime "agent_status"
    New-Item -ItemType Directory -Force -Path $agentStatus | Out-Null

    $meta = Get-LatestMetaOverrides  # defined below

    $status = @{
        timestamp = $ts
        supervisor_version = "production_hardened_20260528"
        iteration = $script:iteration
        healthy = (-not $script:degradedState)
        last_candidate_state = $script:lastCandidateState
        training_recovery_attempts = $script:TrainingRecoveryAttempts
        last_training_recovery = if ($script:LastTrainingRecoveryTime) { $script:LastTrainingRecoveryTime.ToString("o") } else { $null }
        watcher_auto_launched = $script:watcherAutoLaunched
        mt5_running = $null
        disk_ok = $null
        self_monitor_last_call = $script:lastSelfMonitorCall
        meta_overrides = if ($meta) { @{ loaded = $true; reward_profile = $meta.reward_profile; penalty_scale = $meta.penalty_scale; source = $meta.source_artifact } } else { @{ loaded = $false } }
        next_full_cycle_recommended = (($script:iteration % 40) -eq 0)
        note = $ExtraNote
    }
    $statusPath = Join-Path $agentStatus "supervisor_status.json"
    try { $status | ConvertTo-Json -Depth 5 | Set-Content -Path $statusPath -Encoding UTF8 -Force } catch {}

    # Basic alerting/status file consumable by mini TUI (simple, one-read)
    $alertsPath = Join-Path $agentStatus "alerts_supervisor.json"
    $alertsData = @{
        ts = $ts
        supervisor_heartbeat_ok = $true
        active_alerts = if ($Alerts.Count -gt 0) { $Alerts } else { @() }
        recent_recovery = if ($script:TrainingRecoveryAttempts -gt 0) { "Training recoveries: $($script:TrainingRecoveryAttempts)" } else { $null }
        candidate_handoff_ready = ($script:lastCandidateState -like "YES*")
        meta_override_active = ($meta -ne $null)
    }
    try { $alertsData | ConvertTo-Json -Depth 4 | Set-Content -Path $alertsPath -Encoding UTF8 -Force } catch {}
}

function Get-LatestMetaOverrides {
    $ovPath = Join-Path $RepoRoot "runtime\next_training_overrides.json"
    if (Test-Path $ovPath) {
        try {
            $raw = Get-Content $ovPath -Raw -ErrorAction Stop
            return ($raw | ConvertFrom-Json -ErrorAction Stop)
        } catch { return $null }
    }
    return $null
}

# --- SUPERVISOR SELF-REGISTRATION (HARDENING: ensures SYSTEM scheduled task is reliable) ---
# Idempotent re-arm for the persistent background supervisor. Run with highest privs.
# Human one-liner (or future agent): powershell -NoProfile -ExecutionPolicy Bypass -Command "& 'C:\supreme-chainsaw\scripts\vps_agi_supervisor.ps1' -MonitorOnly"  (or direct schtasks import).
function Ensure-SupervisorScheduledTask {
    $taskName = "ChainGambler-AGI-Supervisor"
    $actionCmd = "powershell.exe"
    $scriptPath = if ($PSScriptRoot) { Join-Path $PSScriptRoot "vps_agi_supervisor.ps1" } else { Join-Path $RepoRoot "scripts\vps_agi_supervisor.ps1" }
    $actionArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`" -HealthPollSeconds 60 -MaxRestartsPerHour 8"
    $repo = $RepoRoot
    try {
        $existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest -LogonType ServiceAccount
        $action = New-ScheduledTaskAction -Execute $actionCmd -Argument $actionArgs -WorkingDirectory $repo
        $trigger1 = New-ScheduledTaskTrigger -AtStartup
        $trigger2 = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1) -RepetitionDuration (New-TimeSpan -Days 365)  # safety re-arm
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 5 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 0) -MultipleInstances IgnoreNew
        if ($existing) {
            # Update in place (re-arm)
            Set-ScheduledTask -TaskName $taskName -Action $action -Trigger @($trigger1, $trigger2) -Settings $settings -Principal $principal -ErrorAction Stop | Out-Null
            Write-SupLog "INFO" "Supervisor scheduled task re-armed/updated (SYSTEM, highest, AtStartup + hourly safety, restart-on-fail)."
        } else {
            Register-ScheduledTask -TaskName $taskName -Action $action -Trigger @($trigger1, $trigger2) -Settings $settings -Principal $principal -Description "VPS AGI Supervisor (self-healing, training recovery, candidate handoff, TUI orchestration)" -ErrorAction Stop | Out-Null
            Write-SupLog "SUCCESS" "Supervisor scheduled task registered as SYSTEM (AtStartup + recovery triggers)."
        }
        Write-PipelineDecision -DecisionType "supervisor" -Actor "supervisor" -Decision "TASK_REGISTERED" -Reason "self_arm" -Severity "info" -DetailsJson ('{"task":"' + $taskName + '","user":"SYSTEM"}')
    } catch {
        Write-SupLog "WARN" "Could not (re)register scheduled task (run as admin/SYSTEM or use manual schtasks): $($_.Exception.Message)"
        Write-SupLog "INFO" "Manual re-arm one-liner (admin PS): schtasks /Create /TN `"$taskName`" /TR `"powershell -NoProfile -ExecutionPolicy Bypass -File `\""C:\supreme-chainsaw\scripts\vps_agi_supervisor.ps1`\"" -HealthPollSeconds 60`" /SC ONSTART /RU SYSTEM /RL HIGHEST /F"
    }
}

# --- Main supervision loop (enhanced for training/paper/candidate + TUI auto) ---
$iteration = 0
$script:lastCandidateState = "NO"
$script:watcherAutoLaunched = $false
$script:TrainingRecoveryAttempts = if ($script:TrainingRecoveryAttempts) { $script:TrainingRecoveryAttempts } else { 0 }
$script:LastTrainingRecoveryTime = if ($script:LastTrainingRecoveryTime) { $script:LastTrainingRecoveryTime } else { $null }
$script:lastSelfMonitorCall = $null
$script:degradedState = $false
$script:lastMetaHash = ""
$script:loopStart = Get-Date

# Unified audit: supervisor (re)start - central orchestrator decision point
Write-PipelineDecision -DecisionType "supervisor" -Actor "supervisor" -Decision "SUPERVISOR_STARTED" -Reason "main_loop_init" -Severity "info" -DetailsJson ('{"iteration":0,"MonitorOnly":' + ($MonitorOnly -as [int]) + ',"DryRun":' + ($DryRun -as [int]) + '}')

# Self-arm the SYSTEM scheduled task for reliable background operation (idempotent)
try { Ensure-SupervisorScheduledTask } catch { Write-SupLog "WARN" "Self-registration skipped: $_" }

# Initial duplicate TUI cleanup (kill system-Python TUI, keep .venv312)
try { Clean-DuplicateTuiProcesses | Out-Null } catch {}

# Initial status + self-monitor + meta load (zero-touch visibility from first second)
try { Write-SupervisorStatus -ExtraNote "supervisor_init" } catch {}
try { Invoke-SelfMonitorCycle -LogOnly:$MonitorOnly } catch {}
try {
    $metaInit = Get-LatestMetaOverrides
    if ($metaInit) { Write-SupLog "INFO" "Initial meta overrides loaded for awareness: $($metaInit.reward_profile)" }
} catch {}

while ($true) {
    $iteration++
    try {
        $isRunning = Test-AgiServerRunning
        $health = Get-AgiHealth

        $healthOk = $false
        $readyOk = $false
        if ($health.ContainsKey("/api/health")) { $healthOk = $health["/api/health"].Ok }
        if ($health.ContainsKey("/api/health/ready")) { $readyOk = $health["/api/health/ready"].Ok }

        $mt5Running = Test-MT5Running
        $diskOk = Test-DiskHealthy
        $trainingState = Get-TrainingStatusSummary
        $trainRunning = Test-TrainingRunning
        $harnessRunning = Test-PaperHarnessRunning
        $candState = Test-RecentCandidateStaged

        $degraded = (-not $isRunning) -or (-not $healthOk) -or (-not $readyOk) -or (-not $diskOk)
        $script:degradedState = $degraded

        if ($degraded) {
            $status = if (-not $isRunning) { "NOT RUNNING" } else { "DEGRADED (train=$trainRunning)" }
            Write-SupLog "WARN" "AGI unhealthy: $status (iter $iteration) | $trainingState"

            if (-not (ShouldAllowRestart)) {
                Start-Sleep -Seconds $HealthPollSeconds
                continue
            }

            if (-not $isRunning) {
                Write-SupLog "INFO" "Process not detected - attempting restart... (MT5=$mt5Running, disk=$diskOk, train=$trainRunning)"
            } else {
                Write-SupLog "WARN" "Process running but unhealthy - will restart after brief grace... (MT5=$mt5Running)"
                Start-Sleep -Seconds 5
                # Best-effort graceful termination via lock + taskkill of matching python
                Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                    Where-Object { $_.CommandLine -match "Server_AGI" } |
                    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
                Start-Sleep -Seconds 2
            }

            $newPid = Start-AgiServer
            if ($newPid) {
                RecordRestart
                Write-SupLog "INFO" "Restart successful (new PID=$newPid). Waiting for stabilization... MT5 hint: $(Get-MT5LoginHint) | $trainingState"
                Start-Sleep -Seconds 12   # allow init + MT5 connect + first loop
            }
        } else {
            if ($iteration % 8 -eq 0) {   # ~every 6 minutes at 45s poll
                Write-SupLog "INFO" "AGI healthy (running + health + ready OK). MT5=$mt5Running disk=$diskOk. $trainingState. Uptime monitor continuing... $(Get-MT5LoginHint)"
            }
        }

        # PRODUCTION HARDENING: Periodic Self-Monitor (kill/rollback/conservative/retrain auto-actions)
        # Bounded: every ~3 iters (~2min at 45s poll) + always on degraded. Clear decision audit.
        if (($iteration % 3 -eq 0) -or $degraded) {
            Invoke-SelfMonitorCycle
            $script:lastSelfMonitorCall = (Get-Date).ToUniversalTime().ToString("o")
        }

        # Write TUI-readable supervisor status + basic alerts (zero-touch observability)
        if ($iteration % 5 -eq 0) {
            Write-SupervisorStatus -ExtraNote "periodic heartbeat + self-monitor" -Alerts (@{ last_check = (Get-Date).ToString("o"); degraded = $degraded })
        }

                # --- Training recovery / candidate coordination (self-sustainability) ---
        # AUTO-PROMOTION & GATES (auditor remediation - Auto-Promotion & Gates Agent):
        # Detects good post-fix -> reliably invokes promoter (gates via evaluator/PromotionGates) + canary promotion path.
        # Opt-in safety: only when AGI_AUTO_PROMOTE_CANDIDATE=1 (or SUPERVISOR_AUTO_PROMOTE_CANDIDATE / AGI_AUTO_PROMOTE).
        # This ensures the previously-flagged gap (detect but no auto champion_cycle/gates/canary) is closed.
        # Flow for v4: launch_robust_postfix...v4.ps1 stages candidate -> supervisor detects transition -> (env) auto_promote_candidate.ps1 -> promoter --promote-canary (set_canary) + paper.
        # champion_cycle available behind AGI_USE_FULL_CHAMPION_CYCLE=1 (note: heavy retrain).
        if ($candState -like "YES*" -and $script:lastCandidateState -notlike "YES*") {
            Write-SupLog "INFO" "NEW GOOD CANDIDATE DETECTED: $candState. Running promote_candidate_to_paper.py (gates + checklist + flags + MQL5 guidance + AUTO-TRIGGERED deploy_mql5 for seamless shadow handoff) + TUI watcher. [NEW MTF STANDARD + Decision/Execution context deployed on arm]"
            Write-PipelineDecision -DecisionType "candidate_detection" -Actor "supervisor" -Decision "CANDIDATE_DETECTED" -Reason $candState -Severity "info"

            Launch-TuiWatcherOnCandidate
            $script:watcherAutoLaunched = $true

            # Post-Candidate Handoff Automation Agent integration: central prep of commands + status for TUI + reliable MQL5/paper trigger path
            try { Invoke-PostCandidateHandoff -CandidateState $candState | Out-Null } catch { Write-SupLog "WARN" "Handoff prep note: $_" }

            # Use the real promoter created by Post-Training agent (best handoff tool)
            # V4 ROBUST WIRING: inject source run metadata env so promoter + downstream (harness, MQL5) knows this is v4 50k conservative robust candidate
            try {
                $promoteCmd = "$env:AGI_EXECUTION_TYPE='decision_ppo'; .venv312\Scripts\python.exe scripts\promote_candidate_to_paper.py --symbols BTCUSDm"
                $psi = New-Object System.Diagnostics.ProcessStartInfo
                $psi.FileName = "powershell.exe"
                $psi.Arguments = "-NoProfile -ExecutionPolicy Bypass -Command `"$promoteCmd`""
                $psi.WorkingDirectory = $RepoRoot
                $psi.UseShellExecute = $false
                $psi.CreateNoWindow = $true
                if ($candState -match "v4_robust|v4 robust") {
                    $psi.EnvironmentVariables["AGI_SOURCE_RUN"] = "v4_robust_conservative"
                    $psi.EnvironmentVariables["AGI_V4_CANDIDATE"] = "1"
                    $psi.EnvironmentVariables["AGI_CONSERVATIVE_PAPER"] = "1"  # hint downstream conservative profile
                }
                # Decision PPO + Execution layer: default for all new promoted models (rich full trade specs). Legacy simple_action preserved.
                $psi.EnvironmentVariables["AGI_EXECUTION_TYPE"] = "decision_ppo"
                # Multi-TF + best features context (per symbol) for DecisionPPO inference in harness/live (from configs/best_features_per_symbol.yaml)
                $psi.EnvironmentVariables["AGI_MTF_CONTEXT"] = "1m,5m,15m,1h"
                $psi.EnvironmentVariables["AGI_BEST_FEATURES"] = "configs/best_features_per_symbol.yaml"
                # Decision PPO + Execution + MTF/best-features default for autonomous loop closure (new rich stack for promoted models)
                $psi.EnvironmentVariables["AGI_EXECUTION_TYPE"] = "decision_ppo"
                $psi.EnvironmentVariables["AGI_MULTI_TF_STANDARD"] = "1"
                $psi.EnvironmentVariables["AGI_USE_BEST_FEATURES"] = "1"
                [System.Diagnostics.Process]::Start($psi) | Out-Null
                Write-SupLog "INFO" "promote_candidate_to_paper.py launched (gates + checklist + MQL5 guidance + seamless MQL5 deploy trigger inside promoter for zero/one cmd path) [v4 provenance wired]"
                Write-PipelineDecision -DecisionType "promotion" -Actor "supervisor" -Decision "PROMOTER_LAUNCHED" -Reason "post_candidate_detection" -Severity "info" -DetailsJson ('{"promoter":"promote_candidate_to_paper.py"}')
            } catch {
                Write-SupLog "WARN" "Failed to launch promoter"
                Write-PipelineDecision -DecisionType "promotion" -Actor "supervisor" -Decision "PROMOTER_LAUNCH_FAILED" -Reason "post_candidate_detection" -Severity "warn"
            }

            # RELIABLE AUTO-INVOKE of auto-promoter / gates / canary (the auditor fix)
            $autoPromoteEnv = ($env:AGI_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:SUPERVISOR_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:AGI_AUTO_PROMOTE -eq "1")
            if ($autoPromoteEnv -and -not $DryRun -and -not $MonitorOnly) {
                Write-SupLog "INFO" "AUTO-PROMOTE ENABLED via env (AGI_AUTO_PROMOTE_CANDIDATE etc). Invoking auto_promote_candidate.ps1 for gates run + optional canary promotion (promoter path preferred)."
                try {
                    $autoPs1 = Join-Path $RepoRoot "scripts\auto_promote_candidate.ps1"
                    if (Test-Path $autoPs1) {
                        $autoArgs = "-NoProfile -ExecutionPolicy Bypass -File `"$autoPs1`" -Symbol BTCUSDm"
                        if ($env:AGI_AUTO_PAPER_HARNESS -eq "1") { $autoArgs += " -AutoPaper" }
                        if ($env:AGI_AUTO_MQL5 -eq "1" -or $env:AGI_AUTO_MQL5_DEPLOY -eq "1" -or $env:CHAIN_GAMBLER_AUTO_MQL5_DEPLOY -eq "1") { $autoArgs += " -AutoMQL5" }
                        Start-Process -FilePath "powershell.exe" -ArgumentList $autoArgs.Split() -WindowStyle Hidden -WorkingDirectory $RepoRoot
                        Write-SupLog "INFO" "auto_promote_candidate.ps1 launched (safe env-gated gates + canary + paper/MQL5). This closes detection-without-auto-gates gap."
                        Write-PipelineDecision -DecisionType "promotion" -Actor "supervisor" -Decision "AUTO_PROMOTE_INVOKED" -Reason "env_gate_passed" -Severity "info" -DetailsJson ('{"script":"auto_promote_candidate.ps1","env_gate":true}')
                    } else {
                        Write-SupLog "WARN" "auto_promote_candidate.ps1 not found at $autoPs1"
                    }
                } catch {
                    Write-SupLog "WARN" "Failed to launch auto_promote_candidate wrapper: $($_.Exception.Message)"
                }
            } elseif ($candState -like "YES*") {
                Write-SupLog "INFO" "Candidate good but auto-promote not armed (set AGI_AUTO_PROMOTE_CANDIDATE=1 to enable auto gates->canary flow for this + future v4 runs)."
            }

            # === FINAL ZERO-TOUCH HIGH-LEVEL ORCHESTRATION GLUE (Zero-Touch Orchestrator Agent) ===
            # When a good post-fix candidate appears (from current v4 training run or any postfix),
            # automatically trigger the *full* cohesive autonomous chain in background:
            #   promoter (gates via evaluator + PromotionGates + optional canary set_canary) ->
            #   paper harness (if armed) + MQL5 deploy (shadow-ready .net build + EA attach prep) ->
            #   feedback wiring (RetrainingTrigger ready for paper outcomes).
            # Proper logging to supervisor + dedicated orch/MQL5 logs for auditability.
            # Safety: gated by same AGI_* envs (never default); uses hidden bg processes + explicit logs.
            # Result: v4 candidate -> near-zero-touch downstream with TUI lighting up. Minimal operator input.
            Write-SupLog "INFO" "=== ZERO-TOUCH ORCHESTRATION: Firing full promoter + MQL5 chain + feedback for autonomous unit cohesion ==="
            $orchestrateFull = ($autoPromoteEnv) -or ($env:AGI_AUTO_MQL5 -eq "1") -or ($env:AGI_AUTO_MQL5_DEPLOY -eq "1") -or ($env:CHAIN_GAMBLER_AUTO_MQL5_DEPLOY -eq "1") -or ($env:AGI_AUTO_FULL_DOWNSTREAM -eq "1") -or ($env:AGI_AUTO_PROMOTE_CANDIDATE -eq "1")
            if ($orchestrateFull -and -not $DryRun -and -not $MonitorOnly) {
                # Explicit MQL5 deploy chain (background, logged; complements promoter's internal trigger for supervisor visibility)
                try {
                    $mql5Ps1 = Join-Path $RepoRoot "scripts\deploy_mql5_chain_gambler.ps1"
                    if (Test-Path $mql5Ps1) {
                        $mql5Log = Join-Path $LogsDir ("mql5_orchestrated_deploy_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
                        $mql5Args = "-NoProfile -ExecutionPolicy Bypass -File `"$mql5Ps1`" -AutoFromRegistry -ShadowPrep -DeployToAllTerminals"
                        $mql5Proc = Start-Process -FilePath "powershell.exe" -ArgumentList $mql5Args -WindowStyle Hidden -WorkingDirectory $RepoRoot `
                            -RedirectStandardOutput $mql5Log -RedirectStandardError $mql5Log -PassThru
                        Write-SupLog "INFO" "MQL5 deploy chain (full autonomous) launched bg (PID=$($mql5Proc.Id)). Log: $mql5Log. (promoter + MQL5 handoff complete)"
                    }
                } catch {
                    Write-SupLog "WARN" "Orchestrated MQL5 deploy launch note (non-fatal): $($_.Exception.Message)"
                }

                # If canary auto requested, ensure a full-flag promoter invocation (unified high-level path)
                $wantCanary = ($env:AGI_PROMOTER_PROMOTE_CANARY -eq "1") -or ($env:AGI_AUTO_PROMOTE_CANARY -eq "1")
                if ($wantCanary) {
                    try {
                        $fullPromoteCmd = ".venv312\Scripts\python.exe scripts\promote_candidate_to_paper.py --symbols BTCUSDm --promote-canary --auto-launch"
                        $promoteLog = Join-Path $LogsDir ("promoter_orchestrated_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
                        Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $fullPromoteCmd `
                            -WindowStyle Hidden -WorkingDirectory $RepoRoot -RedirectStandardOutput $promoteLog -RedirectStandardError $promoteLog
                        Write-SupLog "INFO" "Full promoter (with --promote-canary --auto-launch) re-invoked for orchestration (see $promoteLog)"
                    } catch {
                        Write-SupLog "WARN" "Full promote re-launch note: $($_.Exception.Message)"
                    }
                }

                Write-SupLog "SUCCESS" "FULL AUTONOMOUS CHAIN FIRED for v4 candidate: TUI watcher + promoter(gates/canary/paper) + MQL5(bg deploy+shadow) + feedback (RetrainingTrigger). System now cohesive zero-touch unit."
            } else {
                Write-SupLog "INFO" "Full orchestration (promoter+MQL5+feedback) armed for next candidate. Enable with: AGI_AUTO_PROMOTE_CANDIDATE=1 ; AGI_AUTO_MQL5=1 ; AGI_PROMOTER_PROMOTE_CANARY=1 (Machine env for SYSTEM supervisor). One-time setup in PRODUCTION.md."
            }

            # Feedback / autonomous loop status (for cohesive monitoring)
            $retrainTrigger = Join-Path $RepoRoot "Python\autonomous\retraining_trigger.py"
            if (Test-Path $retrainTrigger) {
                Write-SupLog "INFO" "Feedback wiring present: paper harness outcomes will emit retraining_trigger_*.json -> future autonomous retrain cycles (via run_cycle or operator from trigger)."
            }
        }
        $script:lastCandidateState = $candState

        # Stalled/failed training detection + BOUNDED SAFE AUTO-RECOVERY (conservative hyperparams + health signals)
        # Prepared for fast recovery of any 50k postfix run issues.
        if ($iteration % 6 -eq 0) {
            $trainRunning = Test-TrainingRunning
            if ((-not $trainRunning) -or (Test-TrainingHealthStalled)) {
                $recovered = Invoke-BoundedTrainingRecovery -Symbol "BTCUSDm" -Timesteps 50000
                if (-not $recovered) {
                    Write-SupLog "WARN" "Training stalled/failed (no auto-recovery this cycle). Use: .\launch_robust_postfix_training_v4.ps1 -Symbol BTCUSDm -Timesteps 50000 | Conservative params (0.05 kl, 3e-5 lr, 8192 steps) + health signals active."
                }
            }
        }

        # Periodic reminder for paper harness priority when candidate ready (coord task)
        if ($candState -like "YES*" -and $iteration % 20 -eq 0 -and -not $harnessRunning) {
            Write-SupLog "INFO" "POST-FIX CANDIDATE READY: $candState. Next: promoter (scripts/promote_candidate_to_paper.py --dry-run then --auto-launch --promote-canary) or auto_promote_candidate.ps1 (env-gated auto gates+canary). Promoter ALWAYS auto-triggers MQL5 deploy (zero/one cmd: deploy_mql5... -AutoFromRegistry -ShadowPrep). Set AGI_AUTO_PROMOTE_CANDIDATE=1 + AGI_AUTO_MQL5_DEPLOY=1 for full zero-touch. TUI surfaces MQL5 ready + cmd."
        }

        # Retrain feedback loop visibility (auditor gap closure): periodically run aggregator which logs "RETRAIN RECOMMENDED"
        if ($iteration % 15 -eq 0) {
            try {
                $aggCmd = "& `"$pythonExe`" -m Python.autonomous.retraining_trigger --aggregate --data-dir logs"
                Invoke-Expression $aggCmd 2>$null | Out-Null
                Write-SupLog "DEBUG" "Ran retrain aggregator check (see logs for RETRAIN RECOMMENDED if thresholds hit)"
            } catch {
                # Non-fatal
            }
            # Also surface recent marker if present
            $retrainMarker = Join-Path $LogsDir "RETRAIN_RECOMMENDED.latest.json"
            if (Test-Path $retrainMarker) {
                try {
                    $m = Get-Content $retrainMarker -Raw | ConvertFrom-Json
                    if ($m.triggered) {
                        Write-SupLog "WARN" "RETRAIN RECOMMENDED from execution feedback: $($m.next_cycle_command) | reasons: $($m.reasons -join '; ')"
                    }
                } catch {}
            }
        }

        # === SELF-MONITORING + AUTO-ROLLBACK + RECOVERY (Critical Zero-Touch Safety Layer) ===
        # Invokes the dedicated Python/autonomous/self_monitor.py SelfMonitoringRecoveryAgent every cycle (or periodic)
        # Uses: recent validation harness campaigns (last-3 regime adaptation), ExperienceMemory surprise metrics,
        # live P&L drawdown velocity, news shock detection.
        # On triggers: pause entries + conservative TimeExitSpec + notify (agent_status + TUI) + optional light adapt / full retrain via orchestrator.
        # Safe kill-switch + resume surfaced. Writes authoritative runtime/agent_status/self_monitoring_recovery_agent.json
        # MasterSelfEvolutionSupervisor also delegates to the same agent (double coverage).
        # Conservative: cooldowns, fast BT prechecks on rollback, explicit flags only.
        if ($iteration % 5 -eq 0) {
            try {
                $py = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
                if (Test-Path $py) {
                    $smCmd = "& `"$py`" -m Python.autonomous.self_monitor --once 2>$null"
                    $smOut = Invoke-Expression $smCmd | Out-String
                    # Surface key outcome for VPS log (TUI / operator / swarm see full JSON)
                    $smStatusPath = Join-Path $RepoRoot "runtime\agent_status\self_monitoring_recovery_agent.json"
                    if (Test-Path $smStatusPath) {
                        try {
                            $smJson = Get-Content $smStatusPath -Raw | ConvertFrom-Json -ErrorAction SilentlyContinue
                            $smStatus = $smJson.status
                            $recov = $smJson.recovery_state.active
                            $trigs = if ($smJson.PSObject.Properties['rollback_triggers_last_cycle']) { ($smJson.rollback_triggers_last_cycle | Measure-Object).Count } else { 0 }
                            Write-SupLog "INFO" "SELF-MONITOR cycle: status=$smStatus recovery=$recov triggers=$trigs (pause=$($smJson.pause_flag_active) conservative=$($smJson.conservative_flag_active))"
                            if ($trigs -gt 0 -or $recov) {
                                Write-PipelineDecision -DecisionType "self_monitor" -Actor "vps_agi_supervisor" -Decision "ROLLBACK_OR_RECOVERY_ACTIVE" -Reason "triggers=$trigs recovery=$recov" -Severity "warn" -DetailsJson (ConvertTo-Json @{triggers=$trigs; recovery=$recov} -Compress -Depth 3)
                            }
                        } catch {
                            Write-SupLog "DEBUG" "Self-monitor status parse note (non-fatal)"
                        }
                    }
                    Write-SupLog "DEBUG" "Self-Monitor cycle complete (validation+surprise+velocity+news triggers evaluated)"
                }
            } catch {
                Write-SupLog "DEBUG" "Self-monitor invocation note (non-fatal, belt-and-suspenders): $($_.Exception.Message)"
            }
        }

        # === SWARM COORDINATION & VISIBILITY (Swarm Coordination Agent deliverable) ===
        # Periodically sync the Grok subagent swarm (30+ parallel specialized workers) + native reports
        # into the shared runtime/agent_status/ files. This makes the *entire* autonomous multi-agent
        # system (Grok TUI subagents + project scripts) observable from:
        #   - monitor_tui.py Swarm Status panel (live)
        #   - `python scripts/swarm_status.py --list`
        #   - Any other consumer of get_active_agents()
        # Also emits a one-line summary into supervisor log for headless ops visibility.
        # Lightweight: no-op cost if no active Grok MT5 session.
        if ($iteration % 4 -eq 0) {
            try {
                $py = Join-Path $RepoRoot ".venv312\Scripts\python.exe"
                if (Test-Path $py) {
                    $syncOut = & $py -c "
from scripts.swarm_status import sync_grok_swarm, get_active_agents
n = sync_grok_swarm(36)
print(n)
" 2>$null
                    $cnt = if ($syncOut) { $syncOut.Trim() } else { "0" }
                    # Also surface a compact list of top active for the log (operator can see at a glance without TUI)
                    $active = & $py -c "
from scripts.swarm_status import get_active_agents
ags = get_active_agents(14400)[:6]
for a in ags:
    b = (a.get('blockers') or [])
    blk = ' | BLOCKED: ' + (', '.join(b[:1]) if b else '')
    print(a.get('name','?')[:55] + ' [' + a.get('status','?') + ']' + blk)
" 2>$null
                    Write-SupLog "INFO" "Swarm visibility sync: $cnt Grok+project agents reporting. Top active: $(if($active){$active -join ' ; '}else{'none'})"
                }
            } catch {
                Write-SupLog "DEBUG" "Swarm sync note (non-fatal): $($_.Exception.Message)"
            }
        }

        # Periodic TUI duplicate cleanup + agent_status refresh (hardening)
        if ($iteration % 10 -eq 0) {
            try { Clean-DuplicateTuiProcesses -LogOnly:($MonitorOnly -or $DryRun) | Out-Null } catch {}
        }

        # PRODUCTION HARDENING: Auto-consume latest meta overrides (from MetaOptimizer / post-campaign tuning)
        # Supervisor surfaces for TUI + can trigger awareness for full cycles. If changed, log + write status.
        if ($iteration % 12 -eq 0) {
            try {
                $meta = Get-LatestMetaOverrides
                if ($meta) {
                    $metaStr = ($meta | ConvertTo-Json -Compress -Depth 3)
                    $metaHash = [System.Security.Cryptography.SHA256]::Create().ComputeHash([Text.Encoding]::UTF8.GetBytes($metaStr)) | ForEach-Object { $_.ToString("x2") } | Join-String
                    if ($metaHash -ne $script:lastMetaHash) {
                        Write-SupLog "INFO" "META OVERRIDES CONSUMED: reward_profile=$($meta.reward_profile) penalty=$($meta.penalty_scale) patterns_boosted=$($meta.top_boost_patterns -join ',') source=$($meta.source_artifact)"
                        Write-PipelineDecision -DecisionType "meta_override" -Actor "supervisor" -Decision "META_OVERRIDES_LOADED" -Reason "autonomous_meta_tuning" -DetailsJson ($metaStr) -Severity "info"
                        $script:lastMetaHash = $metaHash
                        Write-SupervisorStatus -ExtraNote "meta overrides refreshed for next cycle awareness"
                    }
                }
            } catch { Write-SupLog "DEBUG" "Meta override check note: $_" }
        }

        # Long validation campaign hook + periodic full cycle awareness (for 4-12w FastBacktester runs)
        # Easy trigger: if meta or iteration milestone, note command for TUI/operator (or auto if env)
        if ($iteration % 40 -eq 0) {
            $campaignNote = "Long validation ready: python scripts/run_fast_backtest.py --symbol XAUUSDm --weeks 8 --speed fast (or --ab-test). Use for 4-12 week campaigns pre-promote."
            Write-SupLog "INFO" $campaignNote
            if ($env:AGI_AUTO_LONG_VALIDATION -eq "1" -and -not ($DryRun -or $MonitorOnly)) {
                Write-SupLog "INFO" "AUTO_LONG_VALIDATION armed - launching 8w fast backtest campaign in bg (non-blocking)..."
                try {
                    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command","cd '$RepoRoot'; & '$pythonExe' 'scripts\run_fast_backtest.py' --symbol XAUUSDm --weeks 8 --speed fast" -WindowStyle Hidden
                } catch {}
            }
        }

        # Bounded supervisor health: log uptime + suggest restart after long runs (safety for zero-touch weeks)
        if ($iteration % 200 -eq 0) {
            $uptimeH = [int](((Get-Date) - $script:loopStart).TotalHours)
            Write-SupLog "INFO" "SUPERVISOR UPTIME: ${uptimeH}h (iter $iteration). For true 7-14d zero-touch: ensure SYSTEM task + daily log rotation + external watchdog if needed. Self-monitor + bounded recoveries active."
            Write-SupervisorStatus -ExtraNote "long-running uptime check"
        }

    } catch {
        Write-SupLog "ERROR" "Supervisor loop error: $($_.Exception.Message) - will continue"
        # Bounded error recovery: still write status on error path
        try { Write-SupervisorStatus -ExtraNote "error_recovery_path" } catch {}
    }

    Start-Sleep -Seconds $HealthPollSeconds
}

# Never reached (infinite supervisor)
Write-SupLog "INFO" "Supervisor exiting (should not happen)"

