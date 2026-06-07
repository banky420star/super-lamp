# Auto-promote wrapper (auditor gap fix - Auto-Promotion & Gates Agent)
# Intended to be invoked by vps_agi_supervisor.ps1 (or manually) on detection of good post-fix candidate (alignment_fix_applied).
# 
# SAFE OPT-IN ONLY: Requires AGI_AUTO_PROMOTE_CANDIDATE=1 (or SUPERVISOR_AUTO_PROMOTE_CANDIDATE=1).
# This makes "good candidate detected" -> gates run (via promoter + evaluator + PromotionGates) -> optional canary promotion automatic.
# 
# Prefers lightweight promoter (post-staging eval + paper + optional canary set) over full champion_cycle (which retrains).
# champion_cycle path available via AGI_USE_FULL_CHAMPION_CYCLE=1 (heavy, for full refresh).
#
# Safety rails:
# - Explicit env gate (never auto by default)
# - Respects --dry-run patterns downstream
# - Only acts on post-fix candidates (promoter enforces)
# - Full audit trail + logs
# - For v4/postfix runs: once candidate stages, supervisor with env will trigger this.

param(
    [string]$Symbol = "BTCUSDm",
    [switch]$AutoPaper,
    [switch]$AutoMQL5,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-Location "C:\supreme-chainsaw"

function Write-PipelineDecision {
    param([string]$DecisionType="promotion", [string]$Actor="auto_promote", [string]$Decision, [string]$Candidate="", [string]$RunId="", [string]$Reason="", [string]$DetailsJson="{}", [string]$Severity="info")
    $py = ".\.venv312\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    $argsList = @("-m", "Python.pipeline_audit", "log", "--type", $DecisionType, "--actor", $Actor, "--decision", $Decision, "--reason", $Reason, "--severity", $Severity)
    if ($Candidate) { $argsList += "--candidate", $Candidate }
    if ($RunId) { $argsList += "--run-id", $RunId }
    if ($DetailsJson -and $DetailsJson -ne "{}") { $argsList += "--details-json", $DetailsJson }
    try { & $py $argsList 2>$null | Out-Null } catch {}
    # Also direct append for resilience (matches schema)
    $entry = @{ ts=(Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ"); decision_type=$DecisionType; actor=$Actor; decision=$Decision; candidate=$Candidate; run_id=$RunId; reason=$Reason; details=(ConvertFrom-Json $DetailsJson -EA SilentlyContinue); severity=$Severity } | ConvertTo-Json -Compress
    Add-Content -Path (Join-Path "logs" "PIPELINE_DECISIONS.jsonl") -Value $entry -EA SilentlyContinue
}

# Log entry to unified audit immediately
Write-PipelineDecision -Decision "AUTO_PROMOTE_START" -Reason "wrapper_invoked" -Severity "info"

# --- SAFETY GATE ---
$autoEnabled = ($env:AGI_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:SUPERVISOR_AUTO_PROMOTE_CANDIDATE -eq "1") -or ($env:AGI_AUTO_PROMOTE -eq "1")
if (-not $autoEnabled) {
    Write-Output "SAFETY: AGI_AUTO_PROMOTE_CANDIDATE (or SUPERVISOR_AUTO_PROMOTE_CANDIDATE / AGI_AUTO_PROMOTE) not set to 1. Refusing auto-promotion."
    Write-Output "To enable (opt-in): `$env:AGI_AUTO_PROMOTE_CANDIDATE=`"1`"; .\scripts\auto_promote_candidate.ps1 -Symbol $Symbol"
    Write-Output "Supervisor will auto-invoke this when candidate detected + env set."
    exit 2
}

Write-Output "=== AUTO PROMOTE / GATES CYCLE for $Symbol (OPT-IN ENABLED) ==="
Write-Output "Env gate passed. Flow: candidate -> promoter(gates) -> [canary] -> paper/MQL5"

# Unified PIPELINE_DECISIONS.jsonl (PowerShell direct append for PS1 decision points)
$decFile = "logs\PIPELINE_DECISIONS.jsonl"
$ts = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.fffZ")
$entry = @{ ts=$ts; decision_type="promotion"; actor="auto_promote_ps1"; decision="AUTO_PROMOTE_INVOKED"; candidate=$Symbol; run_id=$Symbol; reason="env_gate"; severity="info" } | ConvertTo-Json -Compress
Add-Content -Path $decFile -Value $entry -Encoding UTF8 -Force -ErrorAction SilentlyContinue

# 1. Prefer promoter (runs real evaluate_candidate_vs_champion + PromotionGates + optional canary set + paper prep)
#    The promoter now supports --promote-canary for direct canary promotion on gate pass (see its docs).
$promoteArgs = "scripts\promote_candidate_to_paper.py --symbols $Symbol --auto-launch"
# Decision PPO + Execution rich stack default for autonomous loop closure (MTF + best_features auto-loaded in harness from configs/)
$env:AGI_EXECUTION_TYPE = "decision_ppo"
# Decision PPO + rich Exec default (MTF + best features)
$env:AGI_EXECUTION_TYPE = if ($env:AGI_EXECUTION_TYPE) { $env:AGI_EXECUTION_TYPE } else { "decision_ppo" }
$env:AGI_MULTI_TF_STANDARD = "1"
$env:AGI_USE_BEST_FEATURES = "1"
if ($DryRun) { $promoteArgs += " --dry-run" }
if ($env:AGI_PROMOTER_PROMOTE_CANARY -eq "1" -or $env:AGI_AUTO_PROMOTE_CANARY -eq "1") {
    $promoteArgs += " --promote-canary"
    Write-Output "AUTO CANARY PROMOTE requested via env/flag"
}
# Decision PPO + Execution layer default for autonomous loop closure (rich specs); MTF/best features context passed via env
if (-not $env:AGI_EXECUTION_TYPE) { $env:AGI_EXECUTION_TYPE = "decision_ppo" }
if (-not $env:AGI_MULTI_TF_STANDARD) { $env:AGI_MULTI_TF_STANDARD = "1" }
if (-not $env:AGI_USE_BEST_FEATURES) { $env:AGI_USE_BEST_FEATURES = "1" }
Write-Output "Execution context: AGI_EXECUTION_TYPE=$($env:AGI_EXECUTION_TYPE) (decision_ppo = rich DecisionPPO+Exec default)"
# V4 ROBUST WIRING: forward source envs so promoter tags artifacts correctly when candidate from launch_robust...v4.ps1
if ($env:AGI_SOURCE_RUN) { $env:AGI_SOURCE_RUN | Out-Null }  # already in env
if ($env:AGI_V4_CANDIDATE -eq "1" -or $env:AGI_V4_ROBUST -eq "1") { Write-Output "v4 robust candidate context forwarded to promoter" }

Write-Output "Invoking promoter for gates + optional canary + harness prep..."
try {
    $py = ".\.venv312\Scripts\python.exe"
    if (-not (Test-Path $py)) { $py = "python" }
    # Arm Decision PPO + rich Execution (default for newly promoted; full trade specs + MTF/best-features). Does not break simple_action legacy.
    $env:AGI_EXECUTION_TYPE="decision_ppo"
    $env:AGI_MTF_CONTEXT="1m,5m,15m,1h"
    $env:AGI_BEST_FEATURES="configs/best_features_per_symbol.yaml"
    & $py $promoteArgs.Split() 
    Write-Output "Promoter completed (gates executed; check logs for strict_promotion_gates + canary decision)."
    Write-PipelineDecision -Decision "AUTO_PROMOTE_PROMOTER_DONE" -Reason "gates_run_complete" -Severity "info"
} catch {
    Write-Output "Promoter invocation warning: $($_.Exception.Message)"
    Write-PipelineDecision -Decision "AUTO_PROMOTE_PROMOTER_FAILED" -Reason $_.Exception.Message -Severity "warn"
}

# 2. Optional full champion_cycle path (re-trains + evaluates; use only when full cycle desired)
$useFullCycle = ($env:AGI_USE_FULL_CHAMPION_CYCLE -eq "1") -or ($env:AGI_USE_FULL_CHAMPION_CYCLE_ON_PROMOTE -eq "1")
if ($useFullCycle) {
    Write-Output "FULL CHAMPION_CYCLE path enabled (heavy retrain+evaluate). Running tools/champion_cycle.py ..."
    try {
        & $py "tools\champion_cycle.py"
        Write-Output "champion_cycle completed (see logs/champion_cycle_last_report.json for wins/passes/strict gates)."
        Write-PipelineDecision -Decision "AUTO_PROMOTE_FULL_CYCLE" -Reason "champion_cycle_used" -Severity "info"
    } catch {
        Write-Output "champion_cycle warning (non-fatal for auto flow): $($_.Exception.Message)"
    }
}

# 3. Optional paper harness (conservative for post-fix)
if ($AutoPaper -or ($env:AGI_AUTO_PAPER_HARNESS -eq "1")) {
    Write-Output "Auto-starting paper harness (conservative) with DecisionPPO+Exec rich specs (default)..."
    try {
        $env:AGI_EXECUTION_TYPE="decision_ppo"
        if (-not $env:AGI_MULTI_TF_STANDARD) { $env:AGI_MULTI_TF_STANDARD="1" }
        if (-not $env:AGI_USE_BEST_FEATURES) { $env:AGI_USE_BEST_FEATURES="1" }
        & $py "scripts\paper_mt5_execution_harness.py" --symbols $Symbol --max-days 7
    } catch {
        Write-Output "Paper harness launch note: $($_.Exception.Message)"
    }
    Write-Output "Decision PPO + Execution stack auto-started in PAPER mode (ExecutionAgent + TradeDecision). Will transition to live on manual supervisor gate after validation. Uses multi-TF context + configs/best_features_per_symbol.yaml per symbol."
}

# 4. MQL5 shadow prep (strengthened for zero-touch Python->MQL5 handoff; promoter now auto-triggers too)
# Promoter (invoked above) ALWAYS handles deploy trigger with good defaults (LogOnly safe or full via AGI_AUTO_MQL5_DEPLOY=1).
# This section remains for direct/legacy or when skipping promoter path.
$autoMql5 = $AutoMQL5 -or ($env:AGI_AUTO_MQL5 -eq "1") -or ($env:AGI_AUTO_MQL5_DEPLOY -eq "1") -or ($env:CHAIN_GAMBLER_AUTO_MQL5_DEPLOY -eq "1")
if ($autoMql5) {
    Write-Output "Triggering MQL5 zero-touch shadow prep (good defaults)..."
    if (Test-Path "scripts\deploy_mql5_chain_gambler.ps1") {
        $mql5Args = "-AutoFromRegistry -ShadowPrep -DeployToAllTerminals"
        if (-not ($env:AGI_AUTO_MQL5_DEPLOY -eq "1" -or $env:CHAIN_GAMBLER_AUTO_MQL5_DEPLOY -eq "1")) {
            $mql5Args += " -LogOnly"
            Write-Output "  (using -LogOnly; set AGI_AUTO_MQL5_DEPLOY=1 for real auto-deploy on promoter/auto flow)"
        }
        & "scripts\deploy_mql5_chain_gambler.ps1" $mql5Args.Split()
        Write-Output "MQL5 deploy complete (check logs/mql5_deploy_*.log + artifacts/mql5_distill/mql5_shadow_ready.json + runtime flag)."
    } else {
        & $py "tools\export_for_mql5.py" --symbol $Symbol
    }
} else {
    Write-Output "MQL5: use one-command (or set env AGI_AUTO_MQL5_DEPLOY=1): .\scripts\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals"
}

Write-Output "Auto promote / gates cycle complete. Review: logs/post_training_promotion_decisions.jsonl + TUI + champion_cycle_last_report.json"
# Production hardening: surface meta + status for TUI (rough edge cleanup)
try {
    $metaFile = "runtime\next_training_overrides.json"
    if (Test-Path $metaFile) {
        $m = Get-Content $metaFile -Raw | ConvertFrom-Json
        Write-Output "META_OVERRIDES_CONSUMED: $($m.reward_profile) penalty=$($m.penalty_scale) (supervisor/TUI visible)"
        @{ts=(Get-Date).ToString("o"); meta=$m; source="auto_promote"} | ConvertTo-Json | Out-File "runtime\agent_status\auto_promote_meta_status.json" -Force
    }
} catch {}
Write-Output "For v4 training benefit: set AGI_AUTO_PROMOTE_CANDIDATE=1 (and optionally AGI_AUTO_MQL5_DEPLOY=1) before starting supervisor; candidate from launch_robust_postfix_training_v4.ps1 will auto-flow: detect (with v4_robust tag) -> promoter (gates+canary+MQL5 auto-prep, with conservative+v4 metadata in artifacts) -> zero/one cmd MQL5 shadow ready."
Write-Output "ZERO-TOUCH MQL5: promoter success now auto-calls the deploy script (env-controlled). TUI surfaces ready state + exact cmd."

# Post-Candidate Handoff Automation marker (for TUI visibility + coordination with supervisor + Current Training Run Monitor)
$rt = "C:\supreme-chainsaw\runtime"
if (-not (Test-Path $rt)) { New-Item -ItemType Directory -Force -Path $rt | Out-Null }
@{ts=(Get-Date).ToString("o"); symbol=$Symbol; auto_env=$autoEnabled; promoter_invoked=$true} | ConvertTo-Json | Out-File -FilePath (Join-Path $rt "auto_promote_last.json") -Encoding UTF8 -Force
Write-Output "Handoff marker: runtime/auto_promote_last.json (TUI Post-Candidate Handoff panel)"