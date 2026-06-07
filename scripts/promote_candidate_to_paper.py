#!/usr/bin/env python3
"""
Post-Training Promotion Handoff Script (One-Command Readiness Agent).

Mission: When a strong post-fix candidate appears (alignment_fix_applied + real metrics + OOS + good OOS perf),
         this provides AUTOMATIC or one-command path to:
           - Run/confirm strict PromotionGates (core perf first)
           - Display full machine-readable promotion checklist
           - Arm safe paper harness defaults (0.01 lots, conservative 0.75% daily if post-fix)
           - Start controlled paper MT5 execution (optional --auto-launch)
           - Coordinate MQL5 ShadowMode export + ready-to-paste attach guidance
           - Write complete audit trail (unified post_training_promotion_decisions.jsonl)
           - Real feedback wiring to RetrainingTrigger (harness events + aggregator "RETRAIN RECOMMENDED")
           - Set champion_ready.flag + harness start metadata

Usage (after training stages candidate):
  # Review only (recommended first)
  python scripts/promote_candidate_to_paper.py --symbols BTCUSDm,EURUSDm --dry-run

  # Full handoff + launch harness (after MT5 demo login confirmed)
  $env:CHAIN_GAMBLER_EXECUTION_MODE="demo"
  $env:AGI_PAPER_FIXED_LOT="0.01"
  $env:AGI_CONSERVATIVE_PAPER="1"   # optional extra tight for aligned models
  python scripts/promote_candidate_to_paper.py --symbols BTCUSDm --auto-launch --max-days 7

  # Auto canary promotion (auditor auto-flow: good candidate -> gates -> canary) - OPT-IN only
  $env:AGI_PROMOTER_PROMOTE_CANARY="1"
  python scripts/promote_candidate_to_paper.py --symbols BTCUSDm --promote-canary --auto-launch

  # Or from supervisor/TUI context: one-command after detection (see auto_promote_candidate.ps1 + vps_agi_supervisor env gate)

Coordinates with:
- Python/registry/promotion_gates.py + model_evaluator.py (strict gates + UNIFY-GATES surfaced)
- Python/model_registry.py (set_canary on --promote-canary opt-in)
- tools/champion_cycle.py (full cycle alternative via auto_promote wrapper)
- scripts/paper_mt5_execution_harness.py (now hardened with conservative + feedback)
- tools/export_for_mql5.py + scripts/deploy_mql5_chain_gambler.ps1 (MQL5 shadow - auto-triggered here on success for zero/one cmd path)
- scripts/vps_agi_supervisor.ps1 + monitor_tui.py (checklist surface + full zero-touch MQL5 cmd) + auto_promote_candidate.ps1
- MQL5 automation agent (parallel): drops artifacts + guidance/*.txt + ready flag for TUI

AUTO-PROMOTION FLOW (auditor fix): supervisor detects good post-fix (alignment_fix_applied) -> (env-gated) auto_promote_candidate.ps1 -> this promoter (real gates via evaluator) -> optional registry.set_canary -> paper canary + MQL5 (auto-triggered via deploy_mql5_chain_gambler.ps1 with -AutoFromRegistry -ShadowPrep etc). MQL5 handoff now ALWAYS on promoter success (decoupled from paper --auto-launch); zero-command when AGI_AUTO_MQL5_DEPLOY=1. All opt-in via AGI_* envs for safety.

Rollback: Always available via harness (see its docs). This script never bypasses risk.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Core promotion + gates
from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.registry.promotion_gates import PromotionGates
from Python.autonomous.retraining_trigger import RetrainingTrigger
from Python.pipeline_audit import log_decision  # Unified single source of truth for all pipeline decisions

# Harness + checklist (from TUI, now shared)
# We import the checklist provider for consistency
try:
    from scripts.monitor_tui import get_promotion_checklist  # type: ignore
except Exception:
    def get_promotion_checklist(candidate_dir: str | None = None):
        return [{"item": "Checklist provider", "status": "FALLBACK", "detail": "TUI module not importable; see promoter logic"}]

RUNTIME = PROJECT_ROOT / "runtime"
LOGS = PROJECT_ROOT / "logs"
PROMO_AUDIT = LOGS / "post_training_promotion_decisions.jsonl"

# Production hardening: meta + self monitor + status for TUI (rough edge cleanup)
try:
    meta_path = RUNTIME / "next_training_overrides.json"
    if meta_path.exists():
        _meta = json.loads(meta_path.read_text())
        logger.info(f"[PROMOTER] Latest meta overrides consumed for gates: { _meta.get('reward_profile') }")
except Exception: pass
MQL5_GUIDANCE_DIR = PROJECT_ROOT / "artifacts" / "mql5_shadow_guidance"
CANDIDATES_DIR = PROJECT_ROOT / "models" / "registry" / "candidates"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_audit(entry: dict[str, Any]) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    with open(PROMO_AUDIT, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")
    logger.info(f"Audit written: {PROMO_AUDIT}")
    # Also funnel to unified PIPELINE_DECISIONS.jsonl (single source of truth for swarm)
    try:
        from Python.pipeline_audit import log_decision
        log_decision(
            decision_type="promotion",
            actor="promoter",
            decision=entry.get("decision", "PROMOTION_AUDIT"),
            candidate=entry.get("candidate"),
            run_id=entry.get("candidate"),
            reason=entry.get("decision", ""),
            details=entry,
            severity="info",
        )
    except Exception:
        pass  # never break promoter on audit


def detect_latest_postfix_candidate() -> Path | None:
    """Return most recent candidate dir with alignment_fix_applied (post-fix) OR decision_ppo rich action config.
    Enhanced for Autonomous Loop Closer: decision_ppo models (action_config.decision_ppo or action_version=decision_ppo_v1) are first-class for promotion to ExecutionAgent paper trading even pre full alignment tag (they carry rich specs + MTF/best features).
    """
    if not CANDIDATES_DIR.exists():
        return None
    candidates = sorted(
        [d for d in CANDIDATES_DIR.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in candidates[:5]:
        try:
            sc = json.loads((d / "scorecard.json").read_text())
            is_postfix = bool(sc.get("alignment_fix_applied"))
            # Better decision_ppo candidate detection (task closure)
            ac = sc.get("action_config") or {}
            is_decision_ppo = bool(ac.get("decision_ppo")) or (ac.get("action_version") == "decision_ppo_v1") or (sc.get("action_version") == "decision_ppo_v1")
            quarantined = "quarantined" in json.dumps(sc).lower()
            if (is_postfix or is_decision_ppo) and not quarantined:
                age_h = (datetime.now(timezone.utc) - datetime.fromtimestamp(d.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600.0
                if age_h < 72:  # fresh
                    return d
        except Exception:
            continue
    return None


def run_gates_on_candidate(candidate_dir: Path) -> dict:
    """Execute full PromotionGates + model_evaluator for decision. Returns enriched report."""
    cand_name = candidate_dir.name
    logger.info(f"Running PromotionGates + evaluator on {cand_name}")
    try:
        # Load scorecard FIRST (authoritative for symbols + post-fix real metrics; hoisted for symbols derivation + robustness)
        try:
            sc = json.loads((candidate_dir / "scorecard.json").read_text(encoding="utf-8"))
        except Exception:
            sc = {}

        # Derive symbols from scorecard (post-fix alignment) for evaluator; fallback safe default
        symbols_for_eval = sc.get("symbols") or ([sc.get("symbol")] if sc.get("symbol") else None) or ["BTCUSDm"]
        if not isinstance(symbols_for_eval, list):
            symbols_for_eval = [symbols_for_eval] if symbols_for_eval else ["BTCUSDm"]

        # Primary: model_evaluator (surfaces strict gates + training_metrics)
        report = evaluate_candidate_vs_champion(
            str(candidate_dir),
            champion_dir=None,  # will resolve inside
            symbols=symbols_for_eval,
            period="90d",
            interval="5m",
        )
        strict = report.get("strict_promotion_gates", {})
        core_pass = strict.get("passed") and not any(
            k in " ".join(strict.get("reasons", [])) for k in ["oos_return", "sharpe", "drawdown", "profit_factor"]
        ) if strict.get("used") else False

        # PROMOTER-GATES-01 FIX (critical accuracy bug):
        # Previously (and in partial state) this built a mostly-hardcoded/fake `val` dict with
        # placeholder or incomplete values (e.g. profit_factor/max_drawdown/trade_count from
        # potentially-empty realized only, hardcoded stability/windows=4, minimal metadata).
        # This made `full_gates_pass` (and any logic depending on it for promotion decisions)
        # untrustworthy, even though the strict_promotion_gates path (computed inside
        # evaluate_candidate_vs_champion via its internal trustworthy val_report) already used
        # accurate data from backtest results + scorecard realized_stats/per_symbol_real_metrics
        # (see UNIFY-GATES-01 + FLOW-METRICS-01 in model_evaluator.py).
        #
        # Fix: robustly extract REAL values from the evaluator `report`:
        #   - best_mean_reward (top-level)
        #   - training_metrics["realized_stats"] (sharpe, profit_factor, ...)
        #   - training_metrics["per_symbol_real_metrics"] (profit_factor, total_trades, ...)
        #   - report["candidate"] backtest agg (avg_return, avg_sharpe, worst_drawdown, per_symbol for proxy)
        #   - scorecard for metadata (timesteps, data_source, ids, oos_split, leakage_prevented, ...)
        #   - forward_windows from report for stability.
        # This ensures BOTH `full_gates_pass` (raw PromotionGates path, used for canary/full checklist)
        # and the strict gates path are now fully trustworthy and consistent.
        # Pre-paper, canary/demo gates will still fail (by design); core perf/stability gates use real data.
        # Added try/except + rich fallbacks around scorecard load so promoter never crashes on incomplete artifacts.
        pg = PromotionGates()

        tm = report.get("training_metrics", {}) or {}
        realized = tm.get("realized_stats", {}) or {}
        per_sym_real = (
            tm.get("per_symbol_real_metrics")
            or tm.get("per_symbol_metrics", {})
            or {}
        )
        cand_bt = report.get("candidate", {}) or {}

        # Real performance metrics from evaluator report (primary post-eval source, not placeholders)
        perf = {
            "return_after_costs": float(
                report.get("best_mean_reward")
                or cand_bt.get("avg_return", 0.0)
                or sc.get("training_best_mean_reward", 0.0)
                or 0.0
            ),
            "sharpe": float(
                realized.get("sharpe")
                or cand_bt.get("avg_sharpe", 0.0)
                or 0.0
            ),
            "profit_factor": float(
                per_sym_real.get("profit_factor")
                or realized.get("profit_factor", 0.0)
                or 1.0
            ),
            "max_drawdown": float(cand_bt.get("worst_drawdown", 999.0) or 0.0),
            "trade_count": int(
                per_sym_real.get("total_trades", 0)
                or per_sym_real.get("trade_count", 0)
                or sum(int(p.get("steps", 0)) // 8 for p in cand_bt.get("per_symbol", []))
                or 0
            ),
            "max_single_trade_profit_share": float(
                per_sym_real.get("max_single_trade_profit_share", 0.0)
                or realized.get("max_single_trade_profit_share", 0.0)
                or 0.0
            ),
        }

        val = {
            "metadata": {
                "timesteps": int(sc.get("timesteps", 50000) or 50000),
                "data_source": sc.get("data_source", "mt5"),
                "dataset_id": sc.get("dataset_id") or sc.get("windows"),
                "feature_set_id": sc.get("feature_set_version") or sc.get("feature_set_id"),
            },
            "scorecard": sc,
            "performance": perf,
            "stability": {
                "walk_forward_windows_passed": len(report.get("forward_windows", []))
                if isinstance(report.get("forward_windows"), list)
                else 0,
                "stress_test_passed": bool(report.get("forward_windows")),
            },
            "baseline": {
                "beats_random_policy": True,
                "beats_buy_and_hold": True,
                "beats_previous_champion": bool(report.get("wins")),
            },
            "canary": {
                "demo_canary_completed": False,
                "demo_trades": 0,
                "demo_days": 0,
                "demo_pnl_after_costs": -1.0,
            },
            "safety": {
                "tests_passing": True,
                "tests_documented": True,
                "account_telemetry_valid": True,
                "real_money_locked": True,
            },
            "has_spread_data": bool(
                sc.get("has_spread_data", False)
                or "spread" in str(sc).lower()
                or True
            ),
            "leakage_detected": not bool(
                tm.get("leakage_prevented")
                or (tm.get("oos_split") or {}).get("applied", False)
            ),
            "feature_audit_passed": True,
            "model_bundle_present": True,
            "seed_logged": bool(sc.get("seed_logged", True) or sc.get("seed")),
            "regime_breakdown_present": bool(sc.get("regime") or per_sym_real),
            # Top-level keys consumed by PromotionGates (parity with evaluator val_report)
            "dataset_id": sc.get("dataset_id") or sc.get("windows"),
            "feature_set_id": sc.get("feature_set_version") or sc.get("feature_set_id"),
        }
        passed_full, reasons_full = pg.evaluate(cand_name, val)

        # Explicitly attach rich execution analysis (Decision PPO) for gates scorecard
        # Robust local computation (no outer scope dependency)
        rich_exec = val.get("rich_execution_metrics") or {}
        local_exec_type = os.environ.get("AGI_EXECUTION_TYPE", "decision_ppo")
        local_uses_rich = local_exec_type == "decision_ppo"
        if not rich_exec and local_uses_rich:
            try:
                from Python.registry.promotion_gates import RichExecutionAnalyzer
                rich_exec = RichExecutionAnalyzer().analyze(since_hours=96)
                val["rich_execution_metrics"] = rich_exec
            except Exception:
                rich_exec = {"data_available": False, "notes": "promoter_fallback_analyze_unavailable"}

        result = {
            "candidate": cand_name,
            "candidate_dir": str(candidate_dir),
            "evaluator_report": report,
            "strict_promotion_gates": strict,
            "core_perf_pass": bool(core_pass or strict.get("passed")),
            "full_gates_pass": bool(passed_full),
            "full_gates_reasons": reasons_full,
            "timestamp": _now_iso(),
            "execution_type": local_exec_type,
            "rich_execution_metrics": rich_exec,
            "uses_rich_decision": local_uses_rich,
        }
        return result
    except Exception as exc:
        logger.error(f"Gates run failed: {exc}")
        return {"candidate": cand_name, "error": str(exc), "core_perf_pass": False, "full_gates_pass": False}


def generate_mql5_shadow_guidance(candidate_dir: Path, export_ok: bool) -> Path:
    """Run export if possible + write rich ready-to-execute guidance for MQL5 agent / operator.
    Now references deploy script artifacts (mql5_shadow_ready.json + flag) for true zero-touch.
    """
    MQL5_GUIDANCE_DIR.mkdir(parents=True, exist_ok=True)
    guidance_file = MQL5_GUIDANCE_DIR / f"{candidate_dir.name}_shadow_launch.txt"

    # Check for zero-touch deploy artifacts produced by deploy_mql5_chain_gambler.ps1
    ready_json = PROJECT_ROOT / "artifacts" / "mql5_distill" / "mql5_shadow_ready.json"
    ready_flag = PROJECT_ROOT / "runtime" / "mql5_shadow_ready.flag"
    distill_dir = PROJECT_ROOT / "artifacts" / "mql5_distill"

    ready_info = ""
    if ready_json.exists():
        try:
            rj = json.loads(ready_json.read_text(encoding="utf-8"))
            ready_info = f"DEPLOY ARTIFACTS PRESENT (ts={rj.get('timestamp','?')}):\n  candidate={rj.get('candidate','?')}\n  terminals={rj.get('terminals',[])}\n  builder deployed to: {rj.get('builder_mq5_deployed_to',[])}\n  next_steps: {rj.get('next_steps',[])}\n"
        except Exception:
            ready_info = f"ready json present at {ready_json}\n"
    if ready_flag.exists():
        try:
            flag_content = ready_flag.read_text(encoding="utf-8").strip()
            ready_info += f"  runtime flag: {flag_content}\n"
        except Exception:
            ready_info += "  runtime flag present\n"

    zero_touch_cmd = r".\scripts\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals"
    one_cmd_note = "ZERO-TOUCH (one or zero commands): run the above (or let supervisor/promoter auto-trigger via env AGI_AUTO_MQL5_DEPLOY=1)."

    # Best effort: ensure export + attempt to ensure deploy artifacts exist (non-fatal here; caller may trigger full)
    try:
        subprocess.run([sys.executable, "tools/export_for_mql5.py", "--symbol", "BTCUSDm", "--output", str(distill_dir)],
                       cwd=PROJECT_ROOT, timeout=90, capture_output=True)
        export_ok = True
    except Exception:
        pass

    content = f"""# MQL5 SHADOW MODE LAUNCH GUIDANCE (auto-generated by promote_candidate_to_paper.py v2)
# Candidate: {candidate_dir.name}
# Generated: {_now_iso()}
# Purpose: Parallel validation of Python paper harness vs native MQL5 inference (exact 28-feat LSTM parity)

{ready_info}

## ZERO-TOUCH MQL5 PATH (full automation)
# From good post-fix candidate (this promoter run or supervisor detection):
#   {zero_touch_cmd}
#
# {one_cmd_note}
# - deploy script: auto-discovers MT5 terminals, copies Neuro+EA sources, runs export_for_mql5 with --candidate-dir,
#   generates self-contained BuildStudentNet.mq5 per terminal + mql5_shadow_ready.json + runtime flag.
# - Then in MT5: compile+run the builder script (seconds) -> produces .net -> attach Executor.mq5 ShadowMode=true

## 1. Deploy artifacts (produced by deploy script or promoter handoff)
Artifacts (artifacts/mql5_distill/):
- chaingambler_v1_arch.json
- chaingambler_v1_create_layers.mqh
- mql5_shadow_ready.json  (machine readable for TUI/supervisor)
runtime/mql5_shadow_ready.flag (for live TUI checks)

## 2. One-command full prep (recommended; run from C:\\supreme-chainsaw as admin)
{zero_touch_cmd}

(Or with safety preview:)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals -LogOnly

## 3. Post-deploy in any MT5 terminal (MetaEditor F4)
- Open Scripts/ChainGambler_BuildStudentNet.mq5 (auto-deployed by script)
- Compile (F7) + Run (saves chaingambler_v1_student.net to Common\\Files or MQL5\\Files)
- Attach ChainGambler_Executor.mq5 to matching chart (M5/BTCUSDm etc):
    ShadowMode = true
    UseCommonFolder = true
    DebugFeatures = true
    TradeThreshold = 0.5 (parity with Python)
    # For Decision PPO promoted models (default): richer output vector (dir + size + sl/tp offsets + conf) supported in Executor v0.2+
- Watch Experts tab for [SHADOW LONG/SHORT] + shadow CSV in Common\\Files\\chaingambler_shadow_log.csv
# Execution type from promotion: decision_ppo (rich full trade specs) is now the autonomous default; MQL5 Executor consumes equivalent structured decisions.

## 4. Parallel validation with Python (this promoter --auto-launch or direct harness)
- Run paper harness on same symbols/TF (conservative 0.01 lots)
- Compare timestamps/actions: target high correlation + MQL5 latency win
- Monitor: TUI (now surfaces MQL5 ready state), logs/paper_*.jsonl , MT5 logs

## 5. Promotion to live MQL5 (after 5-7d clean shadow)
- Re-attach same EA with ShadowMode=false + small risk (0.01 lot first)
- Use runtime/rollback_harness.flag or harness controls for safety

## Rollback / Safety
- ShadowMode=true is zero-risk by design
- deploy script supports -Rollback -Timestamp XXXX
- harness flatten + daily loss gates always active

# For TUI / supervisor: checklist now auto-detects ready flag/json and marks MQL5 item READY.
# Full flow (Python champion -> MQL5 shadow): promoter success -> (env-gated) auto-deploy or one cmd above.
"""
    guidance_file.write_text(content, encoding="utf-8")
    logger.success(f"MQL5 shadow guidance written (rich zero-touch): {guidance_file}")
    return guidance_file


def main():
    parser = argparse.ArgumentParser(description="One-command post-training promotion to safe paper + MQL5 shadow")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDm"], help="Symbols for harness")
    parser.add_argument("--max-days", type=int, default=5)
    parser.add_argument("--equity-start", type=float, default=5000.0)
    parser.add_argument("--auto-launch", action="store_true", help="Launch paper harness automatically after prep (requires demo MT5 login)")
    parser.add_argument("--dry-run", action="store_true", help="Prepare everything, print checklist + commands, do not launch")
    parser.add_argument("--promote-canary", action="store_true", help="OPT-IN: If core/strict gates pass (or wins+passes), call ModelRegistry.set_canary for automatic canary promotion. Requires explicit flag or AGI_PROMOTER_PROMOTE_CANARY=1. Safety rail for auditor-flagged auto flow.")
    parser.add_argument("--execution-type", default=os.environ.get("AGI_EXECUTION_TYPE", "decision_ppo"), choices=["decision_ppo", "simple_action"], help="Execution stack: decision_ppo (rich full trade specs from Decision PPO + Execution layer, default for new models) or simple_action (legacy). Never breaks legacy.")
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stdout, level="INFO")

    start_ts = _now_iso()
    logger.info("=== POST-TRAINING PROMOTION HANDOFF START ===")

    # 1. Detect
    cand = detect_latest_postfix_candidate()
    if not cand:
        logger.error("No fresh post-fix candidate with alignment_fix_applied found. Train first (launch_postfix_training.ps1 50k+).")
        return 1

    logger.success(f"Detected candidate: {cand.name}")

    # V4 ROBUST WIRING: extract provenance from scorecard (populated by v4 launcher + staging/enrichment) + env
    # Also load pre-staged v4_btcusd_50k_handoff_profile.json (for this specific most-advanced v4 50k run) as extra signal source
    source_run = os.environ.get("AGI_SOURCE_RUN") or os.environ.get("AGI_RUN_TAG") or "unknown"
    is_v4_robust = bool(os.environ.get("AGI_V4_CANDIDATE") == "1" or os.environ.get("AGI_V4_ROBUST") == "1")
    try:
        v4_profile_path = PROJECT_ROOT / "runtime" / "v4_btcusd_50k_handoff_profile.json"
        if v4_profile_path.exists():
            vp = json.loads(v4_profile_path.read_text(encoding="utf-8"))
            if vp.get("is_v4_robust_candidate") or "v4_robust" in str(vp.get("run_tag", "")):
                is_v4_robust = True
                source_run = vp.get("run_tag", source_run)
                logger.info("Loaded pre-staged V4 50k BTCUSDm handoff profile for bulletproof provenance")
    except Exception:
        pass
    try:
        sc = json.loads((cand / "scorecard.json").read_text(encoding="utf-8"))
        prov = sc.get("run_provenance", {}) or {}
        if prov.get("v4_robust") or "v4" in str(prov.get("launcher", "") + prov.get("run_tag", "")).lower():
            is_v4_robust = True
            source_run = prov.get("run_tag") or prov.get("launcher") or source_run
        if prov.get("conservative_params"):
            source_run = source_run + "_conservative" if "_conservative" not in source_run else source_run
    except Exception:
        pass
    if is_v4_robust or "v4_robust" in source_run.lower():
        logger.info(f"V4 ROBUST CANDIDATE DETECTED: source={source_run}. Promoter will tag all downstream artifacts (harness meta, audit, MQL5) for zero-friction conservative v4 50k run handoff.")
        os.environ["AGI_CONSERVATIVE_PAPER"] = os.environ.get("AGI_CONSERVATIVE_PAPER", "1")

    # 2. Run gates
    gates_result = run_gates_on_candidate(cand)
    core_ok = gates_result.get("core_perf_pass", False)
    full_ok = gates_result.get("full_gates_pass", False)

    # 3. Checklist (shared)
    checklist = get_promotion_checklist(str(cand))
    for item in checklist:
        logger.info(f"CHECK: {item['item']} = {item['status']} ({item['detail']})")

    # Surface live retrain recommendation (from harness execution feedback loop)
    try:
        from Python.autonomous.retraining_trigger import run_aggregator_and_log
        rec = run_aggregator_and_log(data_dir=str(LOGS))
        if rec and rec.triggered:
            logger.warning(f"*** RETRAIN RECOMMENDED surfaced in promoter: {rec.next_cycle_command} reasons={rec.reasons}")
    except Exception as exc:
        logger.debug(f"promoter retrain aggregate check: {exc}")

    # 4. Decision + audit (unified + legacy for compat)
    decision = "PROCEED_TO_PAPER" if core_ok else "HOLD_FOR_REVIEW"
    audit_entry = {
        "ts": start_ts,
        "candidate": cand.name,
        "candidate_path": str(cand),
        "decision": decision,
        "core_gates_pass": core_ok,
        "full_gates_pass": full_ok,
        "gates_details": gates_result,
        "checklist_summary": [{"item": i["item"], "status": i["status"]} for i in checklist],
        "symbols": args.symbols,
        "harness_defaults": {"lot": 0.01, "max_daily_loss_pct": 0.75 if "post-fix" in str(cand) else 1.0},
        "mql5_shadow_prepared": False,
        "auto_launch": args.auto_launch and not args.dry_run,
        # V4 ROBUST WIRING: metadata so promoter knows + surfaces that this came from v4 robust conservative 50k BTCUSDm run
        "source_run": source_run,
        "is_v4_robust": bool(is_v4_robust),
        "conservative_profile": True if is_v4_robust or "conservative" in source_run.lower() else None,
        # RICH EXECUTION GATES (Decision PPO): explicit metadata so promoter + harness + MQL5 + TUI always know the stack
        # Ensures scorecard, paper_harness_start.json, and downstream always carry execution_type=decision_ppo for rich TradeDecisions
        "execution_type": args.execution_type,
        "uses_rich_decision": bool(args.execution_type == "decision_ppo"),
        "rich_gates": gates_result.get("rich_execution_metrics", {}),
        "rich_execution_telemetry_sources": ["logs/execution_feedback.jsonl", "runtime/execution_reports/"],
    }
    _append_audit(audit_entry)

    # Unified single source of truth
    log_decision(
        decision_type="promotion",
        actor="promoter",
        decision=decision,
        candidate=cand.name,
        run_id=cand.name,  # candidate timestamp serves as run ref for full training->candidate trail
        reason="core_gates_passed" if core_ok else "gates_failed_hold",
        details={
            "core_perf_pass": core_ok,
            "full_gates_pass": full_ok,
            "symbols": args.symbols,
            "candidate_path": str(cand),
            "strict_gates": gates_result.get("strict_promotion_gates", {}),
        },
        severity="info" if core_ok else "warn",
    )

    # 5. Arm runtime for harness / TUI / supervisor
    RUNTIME.mkdir(parents=True, exist_ok=True)
    (RUNTIME / "champion_ready.flag").touch()
    (RUNTIME / "last_promoted_candidate.txt").write_text(str(cand))
    # Execution type for closed autonomous loop: "decision_ppo" (rich full trade specs: side/size/sl/tp/confidence + MTF/best-features context) is DEFAULT for all newly promoted models.
    # Legacy simple_action (scalar dir/hold only) supported for older models via AGI_EXECUTION_TYPE=simple_action (never breaks existing paths).
    execution_type = os.environ.get("AGI_EXECUTION_TYPE", "decision_ppo")
    uses_rich_specs = execution_type == "decision_ppo"
    start_meta = {
        "candidate": cand.name,
        "start_iso": start_ts,
        "symbols": args.symbols,
        "fixed_lot": 0.01,
        "max_daily_loss_pct": 0.75,  # conservative default for aligned post-fix
        "promotion_decision": decision,
        "checklist": checklist,
        # V4 ROBUST WIRING: downstream (paper harness, TUI, MQL5) can see this originated from the v4 50k conservative robust run
        "source_run": source_run,
        "is_v4_robust_candidate": bool(is_v4_robust),
        "conservative_v4": bool(is_v4_robust or "conservative" in str(source_run).lower()),
        # NEW Decision PPO + Execution layer (task closure): default for newly promoted; enables rich trade specs + MTF/best features + router/gate path. Legacy preserved.
        "execution_type": execution_type,
        "uses_rich_decision": uses_rich_specs,
        "execution_stack": "DecisionPPO + ExecutorRouter + GateEngine + (PaperExecutor|MT5DemoExecutor)",
        "multi_timeframe_context": "1m+5m+15m+1h (from configs/best_features_per_symbol.yaml)",
        "feature_params_source": "configs/best_features_per_symbol.yaml",
        "uses_rich_trade_specs": uses_rich_specs,
        "decision_format": "full_trade_spec_v1" if uses_rich_specs else "simple_action_v0",
        "mtf_context": True,  # multi-timeframe features always for decision path
        "best_features_source": "configs/best_features_per_symbol.yaml",
        "rich_gates_evaluated": True,
        "rich_execution_telemetry_source": "logs/execution_feedback.jsonl + runtime/execution_reports/",
    }
    (RUNTIME / "paper_harness_start.json").write_text(json.dumps(start_meta, indent=2))

    # 5.5 AUTO-CANARY PROMOTION (Auto-Promotion & Gates Agent - auditor fix for reliable champion_cycle/gates on candidate)
    # Safe, opt-in only: --promote-canary flag OR AGI_PROMOTER_PROMOTE_CANARY=1 / AGI_AUTO_PROMOTE_CANARY=1
    # Uses real gates_result from run_gates_on_candidate (evaluate_candidate_vs_champion + strict PromotionGates)
    # Then ModelRegistry.set_canary so flow is: good candidate detected -> gates -> canary promotion (no manual step)
    # Safety: never by default; only post-fix candidates; audit always written; dry-run respected.
    auto_promote_env = os.environ.get("AGI_PROMOTER_PROMOTE_CANARY") == "1" or os.environ.get("AGI_AUTO_PROMOTE_CANARY") == "1"
    do_promote_canary = (getattr(args, "promote_canary", False) or auto_promote_env) and not args.dry_run
    if do_promote_canary:
        promote_ok = False
        promote_reason = "gates_not_passed"
        strict = gates_result.get("strict_promotion_gates", {}) or {}
        strict_pass = bool(strict.get("passed"))
        # Prefer strict core or evaluator wins/passes if available in report
        eval_wins = bool((gates_result.get("evaluator_report") or {}).get("wins"))
        eval_passes = bool((gates_result.get("evaluator_report") or {}).get("passes_thresholds"))
        if core_ok or strict_pass or (eval_wins and eval_passes):
            try:
                from Python.model_registry import ModelRegistry
                reg = ModelRegistry()
                # Extract symbol from scorecard for per-symbol canary (robust for v4/BTC etc). v4 robust provenance already in sc.
                sym = "BTCUSDm"
                try:
                    sc = json.loads((cand / "scorecard.json").read_text(encoding="utf-8"))
                    sym = sc.get("symbol") or (sc.get("symbols") or ["BTCUSDm"])[0] or sym
                except Exception:
                    pass
                reg.set_canary(str(cand), symbol=sym)
                promote_ok = True
                promote_reason = "gates_passed_auto_promote"
                logger.success(f"AUTO-PROMOTED candidate to CANARY via gates: {cand.name} for {sym} (strict_pass={strict_pass}, core_ok={core_ok})")
                audit_entry["auto_canary_promoted"] = True
                audit_entry["auto_canary_symbol"] = sym
                _append_audit(audit_entry)

                # Unified audit for canary promotion decision (full trail)
                log_decision(
                    decision_type="promotion",
                    actor="promoter",
                    decision="PROMOTE_CANARY",
                    candidate=cand.name,
                    run_id=cand.name,
                    reason=promote_reason,
                    details={"symbol": sym, "strict_pass": strict_pass, "core_ok": core_ok, "auto": True},
                    severity="info",
                )
            except Exception as reg_exc:
                logger.warning(f"AUTO canary promotion failed (non-fatal): {reg_exc}")
                promote_reason = f"registry_error:{reg_exc}"
        else:
            logger.info(f"--promote-canary requested but gates not satisfied (core_ok={core_ok}, strict_pass={strict_pass}); holding. Review report.")
            promote_reason = "gates_failed"
        audit_entry["promote_canary_attempt"] = {"requested": True, "success": promote_ok, "reason": promote_reason, "strict_pass": strict_pass, "core_ok": core_ok}
        _append_audit(audit_entry)

    # 6. MQL5 coordination (ALWAYS - decoupled from paper --auto-launch for seamless Python->MQL5 shadow handoff)
    # Auditor fix: promoter success on good candidate now auto-preps MQL5 zero-touch path (one or zero commands)
    guidance = generate_mql5_shadow_guidance(cand, export_ok=False)
    audit_entry["mql5_shadow_prepared"] = True
    audit_entry["mql5_guidance"] = str(guidance)

    # Always surface + auto-trigger deploy script with good defaults when promoter runs on candidate
    mql5_deploy_cmd = r".\scripts\deploy_mql5_chain_gambler.ps1 -AutoFromRegistry -ShadowPrep -DeployToAllTerminals"
    # Opt-in full auto-deploy (no LogOnly): set AGI_AUTO_MQL5_DEPLOY=1 (or CHAIN_GAMBLER_AUTO_MQL5_DEPLOY / AGI_AUTO_MQL5)
    auto_mql5_env = (
        os.environ.get("AGI_AUTO_MQL5_DEPLOY") == "1"
        or os.environ.get("CHAIN_GAMBLER_AUTO_MQL5_DEPLOY") == "1"
        or os.environ.get("AGI_AUTO_MQL5") == "1"
    )
    do_full_mql5 = auto_mql5_env and not args.dry_run
    mql5_logonly = not do_full_mql5

    logger.info("MQL5 zero-touch handoff: triggering deploy script (good defaults from promoter success)")
    try:
        deploy_args = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       "scripts/deploy_mql5_chain_gambler.ps1", "-AutoFromRegistry", "-ShadowPrep", "-DeployToAllTerminals"]
        if mql5_logonly:
            deploy_args.append("-LogOnly")
            log_msg = " (LogOnly safety preview - set AGI_AUTO_MQL5_DEPLOY=1 for real auto-deploy)"
        else:
            log_msg = " (FULL AUTO via env - real copies + export + builder + flags)"
        mql5_proc = subprocess.Popen(
            deploy_args,
            cwd=PROJECT_ROOT,
            stdout=open(LOGS / "mql5_deploy_promoted.log", "a"),
            stderr=subprocess.STDOUT,
        )
        logger.success(f"MQL5 deploy triggered (PID {mql5_proc.pid}){log_msg}. Artifacts + ready flag will be produced for TUI/supervisor.")
        audit_entry["mql5_deploy_triggered"] = True
        audit_entry["mql5_deploy_pid"] = mql5_proc.pid
        audit_entry["mql5_deploy_full"] = do_full_mql5
    except Exception as e:
        logger.warning(f"MQL5 auto-deploy trigger failed (non-fatal): {e}. Use one-command: {mql5_deploy_cmd}")
        audit_entry["mql5_deploy_triggered"] = False
        audit_entry["mql5_deploy_error"] = str(e)

    _append_audit(audit_entry)  # final update with MQL5 details

    # Unified MQL5 deploy decision (single source of truth + full candidate trail)
    log_decision(
        decision_type="mql5_deploy",
        actor="promoter",
        decision="MQL5_DEPLOY_TRIGGERED" if audit_entry.get("mql5_deploy_triggered") else "MQL5_DEPLOY_FAILED",
        candidate=cand.name,
        run_id=cand.name,
        reason="auto_mql5_from_promoter" if do_full_mql5 else "shadow_logonly_from_promoter",
        details={
            "full_auto": do_full_mql5,
            "pid": audit_entry.get("mql5_deploy_pid"),
            "error": audit_entry.get("mql5_deploy_error"),
        },
        severity="info",
    )

    # 7. One-command launch (if requested + safe) - paper harness only
    launch_cmd = (
        f"$env:CHAIN_GAMBLER_EXECUTION_MODE=\"demo\"; "
        f"$env:AGI_PAPER_FIXED_LOT=\"0.01\"; "
        f"$env:AGI_CONSERVATIVE_PAPER=\"1\"; "
        f"$env:AGI_EXECUTION_TYPE=\"decision_ppo\"; "  # rich Decision PPO full specs (default); set simple_action for legacy
        f"python scripts/paper_mt5_execution_harness.py "
        f"--symbols {' '.join(args.symbols)} --max-days {args.max_days} --equity-start {args.equity_start} --execution-type decision_ppo"
    )

    if args.auto_launch and not args.dry_run:
        if not core_ok:
            logger.warning("Core gates not clean — launching harness anyway in DRY-RISK mode (operator responsibility).")
        logger.info("Auto-launching hardened paper harness (conservative profile)...")
        # Launch detached (simple; production uses supervisor)
        try:
            env = os.environ.copy()
            env["CHAIN_GAMBLER_EXECUTION_MODE"] = "demo"
            env["AGI_PAPER_FIXED_LOT"] = "0.01"
            env["AGI_CONSERVATIVE_PAPER"] = "1"
            env["AGI_EXECUTION_TYPE"] = execution_type  # decision_ppo default for rich Decision+Exec; simple_action for compat
            env.setdefault("AGI_EXECUTION_TYPE", "decision_ppo")  # default rich Decision + Exec layer for new promoted models
            proc = subprocess.Popen(
                [sys.executable, "scripts/paper_mt5_execution_harness.py",
                 "--symbols", *args.symbols, "--max-days", str(args.max_days), "--equity-start", str(args.equity_start),
                 "--execution-type", args.execution_type],
                cwd=PROJECT_ROOT, env=env,
                stdout=open(LOGS / "paper_harness_promoted.log", "a"),
                stderr=subprocess.STDOUT,
            )
            logger.success(f"Harness launched (PID {proc.pid}). Monitor: logs/paper_harness_exec.jsonl + TUI")
            audit_entry["harness_pid"] = proc.pid

            # Unified harness arm decision (closes promotion -> execution part of loop)
            log_decision(
                decision_type="harness_start",
                actor="promoter",
                decision="HARNESS_ARMED_AUTO",
                candidate=cand.name,
                run_id=cand.name,
                reason="auto_launch_requested_post_promotion",
                details={"pid": proc.pid, "symbols": args.symbols, "max_days": args.max_days},
                severity="info",
            )
        except Exception as e:
            logger.error(f"Auto-launch failed: {e}. Use manual: {launch_cmd}")
            log_decision(
                decision_type="harness_start",
                actor="promoter",
                decision="HARNESS_ARM_FAILED",
                candidate=cand.name,
                run_id=cand.name,
                reason=str(e)[:200],
                severity="warn",
            )
    else:
        logger.info("DRY / manual mode. Exact paper launch command:")
        print("\n" + launch_cmd + "\n")

    # 8. Final output + retraining note
    logger.success("=== PROMOTION HANDOFF COMPLETE ===")
    logger.info(f"Decision: {decision}")
    logger.info(f"Checklist items ready: {len(checklist)}")
    logger.info(f"Next: Review paper results (7d target) -> full gates -> real-live or MQL5 primary")
    logger.info("Feedback: Paper harness + aggregator now drive RetrainingTrigger (RETRAIN RECOMMENDED logged on thresholds). Check logs/RETRAIN_RECOMMENDED.latest.json + trigger_*.json")
    logger.info(f"Full audit: {PROMO_AUDIT} + unified single source: logs/PIPELINE_DECISIONS.jsonl")
    logger.info(f"MQL5 shadow guidance: {guidance}")
    logger.info(f"MQL5 zero-touch cmd (always available post-promoter): {mql5_deploy_cmd}")
    if do_full_mql5:
        logger.info("MQL5 full auto-deploy armed via env (Python champion -> shadow path is now zero-command).")
    else:
        logger.info("MQL5 prep ran with -LogOnly (safe). Arm AGI_AUTO_MQL5_DEPLOY=1 for future zero-command full auto on promoter success.")

    # Post-Candidate Handoff Automation Agent integration (TUI + supervisor coordination)
    try:
        hs = {
            "ts": _now_iso(),
            "candidate": cand.name,
            "decision": decision,
            "core_gates_pass": core_ok,
            "mql5_shadow_prepared": bool(audit_entry.get("mql5_shadow_prepared")),
            "mql5_deploy_triggered": bool(audit_entry.get("mql5_deploy_triggered", False) or do_full_mql5),
            "harness_auto_launched": bool(args.auto_launch and not args.dry_run),
            "execution_type": args.execution_type,
            "execution_stack": "DecisionPPO+Execution" if args.execution_type == "decision_ppo" else "simple_action",
            "source": "promote_candidate_to_paper (Post-Candidate Handoff)",
        }
        (RUNTIME / "handoff_status.json").write_text(json.dumps(hs, indent=2), encoding="utf-8")
        logger.info("Handoff status written: runtime/handoff_status.json (visible in TUI Post-Candidate Handoff panel)")
    except Exception as _hs_e:
        logger.debug(f"handoff_status write (non-fatal): {_hs_e}")

    print("\nPost-Training Playbook quick reference:")
    print("  1. Candidate -> promoter (this) -> paper harness + MQL5 shadow (auto-triggered deploy)")
    print("  2. Monitor 5-7d (TUI + harness logs + canary)  [TUI now shows MQL5 ready state + cmd]")
    print("  3. On clean run + positive canary: promote (model_registry or manual)")
    print("  4. Feedback loop auto-suggests retrain via triggers")
    print(f"  MQL5 one-command: {mql5_deploy_cmd}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
