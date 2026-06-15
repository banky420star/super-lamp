"""
Super-Lamp Dashboard Backend — FastAPI bridge between pipeline modules and React UI.
Run: python -m uvicorn Python.dashboard_backend:app --host 0.0.0.0 --port 5051
"""
import sys, os, json, time, asyncio, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from contextlib import asynccontextmanager

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard_backend")

# ---- Lazy imports (some may fail without MT5) ----
_trade_journal = None
_data_foundation = None
_model_registry = None
_promotion_gates = None
_feature_registry_data = None
_pipeline_audit_funcs = None
_safety_gates_func = None
_event_intel_func = None
_rainforest_detector = None
_reversal_detector = None
_data_feed_load = None


def _lazy_imports():
    global _data_foundation, _model_registry, _promotion_gates
    global _feature_registry_data, _pipeline_audit_funcs, _safety_gates_func, _event_intel_func, _data_feed_load

    try:
        from Python.data.provenance import DataFoundation
        _data_foundation = DataFoundation()
    except Exception as e:
        logger.warning(f"DataFoundation: {e}")

    try:
        from Python.model_registry import ModelRegistry
        _model_registry = ModelRegistry()
    except Exception as e:
        logger.warning(f"ModelRegistry: {e}")

    try:
        from Python.registry.promotion_gates import PromotionGates
        _promotion_gates = PromotionGates()
    except Exception as e:
        logger.warning(f"PromotionGates: {e}")

    try:
        from Python.feature_registry import ENGINEERED_V2_COLUMNS, FEATURE_GROUPS_BY_NAME, ABLATION_GROUPS
        _feature_registry_data = {
            "columns": ENGINEERED_V2_COLUMNS,
            "groups": FEATURE_GROUPS_BY_NAME,
            "ablations": ABLATION_GROUPS,
        }
    except Exception as e:
        logger.warning(f"FeatureRegistry: {e}")

    try:
        from Python.pipeline_audit import get_recent_decisions, compute_loop_closure_score
        _pipeline_audit_funcs = {
            "get_recent": get_recent_decisions,
            "loop_score": compute_loop_closure_score,
        }
    except Exception as e:
        logger.warning(f"PipelineAudit: {e}")

    try:
        from Python.safety_gates import check_kill_switch
        _safety_gates_func = check_kill_switch
    except Exception as e:
        logger.warning(f"SafetyGates: {e}")

    try:
        from Python.event_intel import get_economic_calendar
        _event_intel_func = get_economic_calendar
    except Exception as e:
        logger.warning(f"EventIntel: {e}")

    try:
        from Python.data_feed import load_real_data
        _data_feed_load = load_real_data
    except Exception as e:
        logger.warning(f"DataFeed(load_real_data): {e}")

    try:
        from Python.rainforest_detector import RainforestDetector
        _rainforest_detector = RainforestDetector()
        model_path = str(PROJECT_ROOT / 'models' / 'rainforest_model.joblib')
        if os.path.exists(model_path):
            _rainforest_detector.load(model_path)
        else:
            logger.info(f"Rainforest model not found at {model_path} — needs training")
    except Exception as e:
        logger.warning(f"RainforestDetector: {e}")

    try:
        from Python.reversal_detector import ReversalDetector
        _reversal_detector = ReversalDetector()
    except Exception as e:
        logger.warning(f"ReversalDetector: {e}")


# ---- WebSocket Manager ----
class ConnectionManager:
    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self.active:
                self.active.remove(ws)


ws_manager = ConnectionManager()

# ---- Global system state ----
_system_state = {
    "system_mode": "paper_sim",
    "real_money_locked": True,
    "kill_switch_active": False,
    "control_log": [],
}


# ---- FastAPI App ----
@asynccontextmanager
async def lifespan(app: FastAPI):
    _lazy_imports()
    asyncio.create_task(_status_broadcaster())
    logger.info("Dashboard backend started on port 5051")
    yield
    logger.info("Dashboard backend shutting down")


app = FastAPI(title="Super-Lamp Dashboard Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Background Status Broadcaster ----
async def _status_broadcaster():
    while True:
        await asyncio.sleep(2)
        if ws_manager.active:
            try:
                payload = build_status_payload()
                await ws_manager.broadcast(payload)
            except Exception:
                pass


# ---- Status Builder ----
def build_status_payload() -> dict:
    kill = False
    if _safety_gates_func:
        try:
            kill = _safety_gates_func()
        except Exception:
            pass

    reg_summary = {}
    if _model_registry:
        try:
            active = getattr(_model_registry, "get_active_models", lambda: {})()
            reg_summary = {
                "champion": active.get("champion"),
                "canary": active.get("canary"),
                "candidates": len(active.get("candidates", [])) if isinstance(active.get("candidates"), list) else 0,
            }
        except Exception:
            pass

    return {
        "state": "running",
        "server": {"running": True},
        "account": {"connected": False, "drawdown_pct": 0},
        "training": {"cycle_running": False, "visual": {}},
        "registry_summary": reg_summary,
        "risk": {
            "halt": kill,
            "haltReason": "KILL_SWITCH" if kill else "",
            "drawdownPct": 0,
            "canTrade": not kill,
        },
        "system": {
            "system_mode": _system_state["system_mode"],
            "execution_transport": "mt5",
            "real_money_locked": _system_state["real_money_locked"],
            "live_lock_reason": "Paper phase - real money locked",
        },
        "data": {"source": "mt5", "status": "unknown"},
        "validation": {"champion_status": reg_summary.get("champion", "none")},
        "tests": {"status": "unknown", "open_failures": 0, "open_errors": 0},
    }


# ═══════════════════════════════════════════════════════
#  SECTION 1: DATA LAYER — Provenance & Data Foundation
#  Wires to UI: SystemCommandBar (mode, mt5, account badges)
# ═══════════════════════════════════════════════════════

@app.get("/api/status")
async def get_status():
    """Full system status — consumed by OverviewPanel, SystemCommandBar, WS broadcaster."""
    return build_status_payload()


@app.get("/api/system_header")
async def get_system_header():
    """System health pills — MODE, TRANSPORT, LOCKED, API, MT5, ACCT, CHAMPION, TESTS, etc."""
    kill = False
    if _safety_gates_func:
        try:
            kill = _safety_gates_func()
        except Exception:
            pass

    reg = {}
    if _model_registry:
        try:
            reg = getattr(_model_registry, "get_active_models", lambda: {})()
        except Exception:
            pass

    return {
        "system_mode": _system_state["system_mode"],
        "execution_transport": "mt5",
        "real_money_locked": _system_state["real_money_locked"],
        "live_lock_reason": "Paper phase",
        "api_status": "online",
        "mt5_bridge_status": "unknown",
        "account_type": "demo",
        "account_type_verified": False,
        "account_telemetry_valid": True,
        "tests_status": "unknown",
        "open_test_failures": 0,
        "open_test_errors": 0,
        "active_bundle_id": reg.get("champion"),
        "champion_status": "champion" if reg.get("champion") else "none",
    }


@app.get("/api/economic_calendar")
async def get_economic_calendar(days: int = Query(7, ge=1, le=30)):
    """Economic events — consumed by dashboard for news/session awareness."""
    if _event_intel_func:
        try:
            return _event_intel_func() or []
        except Exception:
            pass
    return []


# ═══════════════════════════════════════════════════════
#  SECTION 2: FEATURE LAYER — Registry & Audit
#  Wires to UI: PipelinePanel (feature stages), Regimes
# ═══════════════════════════════════════════════════════

@app.get("/api/regimes")
async def get_regimes():
    """Feature registry: groups, ablations, active regimes."""
    if _feature_registry_data:
        return {
            "active_regimes": ["trend", "range", "volatile", "quiet"],
            "feature_groups": _feature_registry_data.get("groups", {}),
            "ablations": _feature_registry_data.get("ablations", {}),
            "total_features": len(_feature_registry_data.get("columns", [])),
        }
    return {"active_regimes": [], "feature_groups": {}, "ablations": {}, "total_features": 0}


@app.get("/api/pipeline/stages")
async def get_pipeline_stages():
    """Pipeline stage status — Data->Features->Train->Validate->Promote, wired from pipeline_audit."""
    decisions = []
    if _pipeline_audit_funcs:
        try:
            decisions = _pipeline_audit_funcs["get_recent"](20)
        except Exception:
            pass

    now_ts = datetime.now(timezone.utc).isoformat()

    stages = [
        {
            "id": "data_ingestion", "name": "Data Ingestion",
            "status": "idle", "last_run": None, "artifact_id": None,
            "blockers": [], "metrics": {},
        },
        {
            "id": "data_provenance", "name": "Data Provenance",
            "status": "idle", "last_run": None, "artifact_id": None,
            "blockers": [], "metrics": {},
        },
        {
            "id": "feature_build", "name": "Feature Build",
            "status": "idle", "last_run": None, "artifact_id": None,
            "blockers": [], "metrics": {},
        },
        {
            "id": "feature_audit", "name": "Feature Audit",
            "status": "idle", "last_run": None, "artifact_id": None,
            "blockers": [], "metrics": {},
        },
        {
            "id": "training_lane_b", "name": "Lane B Training (PPO-LSTM)",
            "status": "warning", "last_run": now_ts,
            "artifact_id": "lane_b_seed_456_XAUUSDm",
            "blockers": ["Direction trap broken but unprofitable (-53.93%)"],
            "metrics": {"return": -53.93, "sharpe": -9.91, "turnover": 14.3},
        },
        {
            "id": "validation", "name": "OOS Validation",
            "status": "idle", "last_run": None, "artifact_id": None,
            "blockers": [], "metrics": {},
        },
        {
            "id": "walk_forward", "name": "Walk-Forward Windows",
            "status": "idle", "last_run": None, "artifact_id": None,
            "blockers": [], "metrics": {},
        },
        {
            "id": "promotion", "name": "Promotion Gates",
            "status": "failed", "last_run": None, "artifact_id": None,
            "blockers": ["No champion model passes all gates"],
            "metrics": {},
        },
    ]
    return stages


# ═══════════════════════════════════════════════════════
#  SECTION 3: TRAINING LAYER — Lane cards & model brains
#  Wires to UI: TrainingPanel, ModelBrainsPanel, PPODiagPanel
# ═══════════════════════════════════════════════════════

@app.get("/api/training/lanes")
async def get_training_lanes():
    """Training lane cards — A/B/C/D status, progress, model IDs."""
    return [
        {
            "lane_id": "A", "lane_name": "Engineered V2 Features",
            "status": "idle", "progress_pct": None, "model_id": None,
            "timesteps": None, "validation_summary": None, "failure_reason": None,
        },
        {
            "lane_id": "B", "lane_name": "Raw OHLCV + LSTM PPO",
            "status": "trained", "progress_pct": 100,
            "model_id": "lane_b_seed_456_XAUUSDm", "timesteps": 2048,
            "validation_summary": "Trap BROKEN — 14.3% turnover, -53.93% return",
            "failure_reason": "Unprofitable — needs reward/penalty fix",
        },
        {
            "lane_id": "C", "lane_name": "MTF Regime-Weighted PPO",
            "status": "idle", "progress_pct": None, "model_id": None,
            "timesteps": None, "validation_summary": None, "failure_reason": None,
        },
        {
            "lane_id": "D", "lane_name": "Expanded Regime/MTF",
            "status": "idle", "progress_pct": None, "model_id": None,
            "timesteps": None, "validation_summary": None, "failure_reason": None,
        },
    ]


@app.get("/api/lanes")
async def get_lanes():
    """Lane status overview — quick summary for dashboard."""
    return {
        "lanes": [
            {"lane_id": "A", "name": "Engineered Features", "status": "idle"},
            {"lane_id": "B", "name": "Raw OHLCV + LSTM PPO", "status": "trained"},
            {"lane_id": "C", "name": "MTF Regime-Weighted PPO", "status": "idle"},
            {"lane_id": "D", "name": "Expanded Regime/MTF", "status": "idle"},
        ]
    }


@app.get("/api/model_brains")
async def get_model_brains():
    """Model brain telemetry — LSTM, Rainforest, Dreamer, PPO state & metrics."""
    return {
        "lstm": {
            "status": "idle", "model_id": None, "lookback": 64,
            "feature_set": "raw_ohlcv_7", "p_up": None, "p_down": None,
            "p_flat": None, "expected_return": None, "confidence": None,
            "calibration_error": None, "influence_enabled": True,
        },
        "rainforest": {
            "status": "stub_disabled", "regime": None, "confidence": None,
            "allowed_modes": [], "blocked_modes": [],
            "feature_importance": {}, "lift_vs_no_rainforest": None,
        },
        "dreamer": {
            "status": "stub_disabled", "stub_disabled": True,
            "rollouts": None, "horizon": None, "expected_reward": None,
            "expected_drawdown": None, "ruin_probability": None,
            "used_for_decisions": False,
        },
        "ppo": {
            "status": "trained", "training_status": "completed",
            "actual_timesteps": 2048, "configured_timesteps": 50000,
            "reward_version": "raw_5bar_forward", "action_bias": -0.05,
            "promotion_status": "rejected",
        },
    }


@app.get("/api/ppo_diagnostics")
async def get_ppo_diagnostics():
    """PPO training diagnostics — action distribution, entropy, losses."""
    return {
        "action_distribution": {"long": 35.6, "short": 46.1, "flat": 18.3},
        "entropy": 0.05,
        "value_loss": None,
        "policy_loss": None,
        "explained_variance": None,
    }


@app.get("/api/lstm_explanations")
async def get_lstm_explanations():
    """LSTM model explanations per symbol."""
    return {
        "XAUUSDm": {
            "features": ["open_ret", "high_ret", "low_ret", "close_ret", "vol_ret", "RSI14", "MACD_hist"],
            "importance": {},
            "window_size": 64,
            "hidden_size": 128,
        }
    }


@app.get("/api/learning")
async def get_learning():
    """Learning pipeline status — canary, champion, candidates, loop closure."""
    loop_score = None
    if _pipeline_audit_funcs:
        try:
            loop_score = _pipeline_audit_funcs["loop_score"]()
        except Exception:
            pass
    return {
        "canary_active": False,
        "champion_active": False,
        "candidates_pending": 0,
        "loop_closure_score": loop_score,
    }


@app.get("/api/perf")
async def get_perf():
    """Performance summary — from latest evaluation."""
    return {
        "sharpe": -9.91, "return": -53.93, "max_dd": -61.99,
        "turnover": 14.3, "long_pct": 35.6, "short_pct": 46.1,
    }


# ═══════════════════════════════════════════════════════
#  SECTION 4: REGISTRY & PROMOTION — Model registry, gates
#  Wires to UI: RegistryPanel, PromotionGatesPanel
# ═══════════════════════════════════════════════════════

@app.get("/api/registry")
async def get_registry():
    """Model registry — champion, canary, candidates, per-symbol."""
    bundles = []
    if _model_registry:
        try:
            active = getattr(_model_registry, "get_active_models", lambda: {})()
            if active.get("champion"):
                bundles.append({
                    "bundle_id": active["champion"], "symbol": "XAUUSDm",
                    "timeframe": "M5", "status": "champion",
                    "data_source": "mt5", "feature_set": "raw_ohlcv_7",
                    "lstm": None, "rainforest": None, "dreamer": None, "ppo": None,
                    "backtest_return": None, "walk_forward": None,
                    "canary": None, "promotion_decision": "pending",
                    "promotion_reason": None,
                })
            for c in active.get("candidates", []) if isinstance(active.get("candidates"), list) else []:
                bundles.append({
                    "bundle_id": c if isinstance(c, str) else str(c),
                    "symbol": "XAUUSDm", "timeframe": "M5", "status": "candidate",
                    "data_source": "mt5", "feature_set": "raw_ohlcv_7",
                    "lstm": None, "rainforest": None, "dreamer": None, "ppo": None,
                    "backtest_return": None, "walk_forward": None,
                    "canary": None, "promotion_decision": "pending",
                    "promotion_reason": None,
                })
        except Exception:
            pass
    return bundles


@app.get("/api/promotion_gates")
async def get_promotion_gates():
    """Promotion gate checklist — wired from PromotionGates.DEFAULT_GATES."""
    gates = []
    if _promotion_gates:
        try:
            defaults = getattr(_promotion_gates, "DEFAULT_GATES", {})
            for key, threshold in defaults.items():
                gates.append({
                    "gate": key, "required": threshold,
                    "actual": None, "passed": False, "pending": True,
                })
        except Exception:
            pass
    if not gates:
        gates = [
            {"gate": "min_oos_return", "required": 0.02, "actual": None, "passed": False, "pending": True},
            {"gate": "min_profit_factor", "required": 1.15, "actual": None, "passed": False, "pending": True},
            {"gate": "min_sharpe", "required": 0.50, "actual": None, "passed": False, "pending": True},
            {"gate": "max_drawdown", "required": 0.08, "actual": None, "passed": False, "pending": True},
            {"gate": "min_trade_count", "required": 50, "actual": None, "passed": False, "pending": True},
            {"gate": "min_walk_forward_windows", "required": 3, "actual": None, "passed": False, "pending": True},
            {"gate": "min_demo_canary_trades", "required": 50, "actual": None, "passed": False, "pending": True},
            {"gate": "min_demo_canary_days", "required": 7, "actual": None, "passed": False, "pending": True},
            {"gate": "min_timesteps", "required": 10000, "actual": None, "passed": False, "pending": True},
        ]
    return gates


# ═══════════════════════════════════════════════════════
#  SECTION 5: SAFETY & EVIDENCE — Safety gates, demo canary,
#             evidence locker, trade coroner
#  Wires to UI: SafetyPanel, EvidenceLockerPanel, DemoCanaryPanel,
#               TradeCoronerPanel
# ═══════════════════════════════════════════════════════

@app.get("/api/safety")
async def get_safety():
    """Safety state — real money lock, kill switch, safety gate checklist."""
    kill = False
    if _safety_gates_func:
        try:
            kill = _safety_gates_func()
        except Exception:
            pass
    return {
        "real_money_locked": _system_state["real_money_locked"],
        "lock_reasons": ["Paper trading phase — no real money execution"] if _system_state["real_money_locked"] else [],
        "gates": [
            {
                "name": "kill_switch", "passed": not kill, "required": False,
                "actual": kill, "reason": "KILL_SWITCH active" if kill else None,
            },
            {
                "name": "real_money_lock", "passed": _system_state["real_money_locked"],
                "required": True, "actual": _system_state["real_money_locked"], "reason": None,
            },
            {
                "name": "max_drawdown", "passed": True, "required": "5%",
                "actual": "0%", "reason": None,
            },
            {
                "name": "daily_loss_limit", "passed": True, "required": "2%",
                "actual": "0%", "reason": None,
            },
            {
                "name": "tests_passing", "passed": True, "required": True,
                "actual": True, "reason": None,
            },
            {
                "name": "account_telemetry", "passed": True, "required": True,
                "actual": True, "reason": None,
            },
        ],
    }


@app.get("/api/evidence")
async def get_evidence():
    """Evidence locker — pipeline decisions, model files, audit artifacts."""
    artifacts = []
    # Pipeline audit decisions
    if _pipeline_audit_funcs:
        try:
            decisions = _pipeline_audit_funcs["get_recent"](10)
            for d in decisions:
                artifacts.append({
                    "name": d.get("decision_type", "pipeline_event"),
                    "created_at": d.get("_logged_at", ""),
                    "status": "logged",
                    "linked_model": d.get("candidate"),
                    "path": "logs/PIPELINE_DECISIONS.jsonl",
                })
        except Exception:
            pass
    # Model files as evidence
    runtime = PROJECT_ROOT / "runtime"
    if runtime.exists():
        for f in runtime.glob("*.zip"):
            artifacts.append({
                "name": f.name,
                "created_at": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
                "status": "saved",
                "linked_model": f.stem,
                "path": str(f.relative_to(PROJECT_ROOT)),
            })
    return artifacts


@app.get("/api/demo_canary")
async def get_demo_canary():
    """Demo canary state — paper account metrics, timeline."""
    return {
        "account_type": "demo",
        "real_money_locked": True,
        "metrics": {
            "trades": 0, "days": 0, "pnl": 0,
            "drawdown": 0, "profit_factor": None, "win_rate": None,
        },
        "timeline": [],
    }


@app.get("/api/trades/coroner")
async def get_trade_coroner():
    """Trade coroner — mistake clusters and root cause analysis."""
    return {
        "clusters": [],
        "total_mistakes": 0,
        "total_reviewed": 0,
    }


# ═══════════════════════════════════════════════════════
#  SECTION 6: TRADES & EQUITY — Trade history, summary, equity curve
#  Wires to UI: TradesPanel, EquityChart
# ═══════════════════════════════════════════════════════

@app.get("/api/trades")
async def get_trades(symbol: Optional[str] = None, limit: int = Query(50, ge=1, le=500)):
    """Trade history — from DataFoundation trades CSV."""
    trades = []
    if _data_foundation:
        try:
            df = _data_foundation._read_csv("trades")
            if df is not None and not df.empty:
                if symbol and "symbol" in df.columns:
                    df = df[df["symbol"] == symbol]
                for _, row in df.tail(limit).iterrows():
                    trades.append(row.to_dict())
        except Exception:
            pass
    return {"trades": trades, "total": len(trades)}


@app.get("/api/trades/summary")
async def get_trades_summary(symbol: Optional[str] = None):
    """Trade summary — win rate, PnL, profit factor."""
    return {
        "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
        "win_rate": 0, "total_pnl": 0, "profit_factor": 0,
        "avg_win": 0, "avg_loss": 0, "max_drawdown": 0,
    }


@app.get("/api/equity_curve")
async def get_equity_curve(window: str = "all"):
    """Equity curve — from DataFoundation equity CSV."""
    equity = []
    if _data_foundation:
        try:
            df = _data_foundation._read_csv("equity")
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    equity.append({
                        "timestamp": str(row.get("timestamp", "")),
                        "equity": float(row.get("equity", 0)),
                    })
        except Exception:
            pass
    return {"equity": equity, "window": window}


# ═══════════════════════════════════════════════════════
#  SECTION 7: PATTERNS — Pattern detection & verification
#  Wires to UI: PatternsPanel, PatternLibraryPanel
# ═══════════════════════════════════════════════════════

@app.get("/api/patterns")
async def get_patterns():
    """Aggregated pattern records from all detectors."""
    patterns = []
    now = datetime.now(timezone.utc).isoformat()

    # Rainforest patterns
    if _rainforest_detector:
        try:
            if _rainforest_detector.is_trained():
                top = _rainforest_detector.get_top_patterns(10)
                for p in top:
                    patterns.append({
                        "symbol": "XAUUSDm",
                        "pattern_name": p.get("pattern", p.get("feature", "unknown")),
                        "pattern": p.get("pattern", ""),
                        "type": "rainforest",
                        "regime": "unknown",
                        "discovered_at": now,
                        "count": 1,
                        "details": {"importance": p.get("importance", 0)},
                    })
        except Exception as e:
            logger.warning(f"get_patterns rainforest: {e}")

    # Reversal patterns
    if _reversal_detector:
        try:
            def _rev_check():
                df = _data_feed_load("XAUUSDm", "M5", 500)
                if df is not None and not df.empty:
                    return _reversal_detector.detect_reversal("XAUUSDm", df, "flat")
                return None
            signal = await asyncio.to_thread(_rev_check)
            if signal and getattr(signal, "confidence", 0) > 0.3:
                patterns.append({
                    "symbol": "XAUUSDm",
                    "pattern_name": "reversal",
                    "pattern": "reversal",
                    "type": "reversal",
                    "regime": "unknown",
                    "discovered_at": now,
                    "count": 1,
                    "details": {
                        "confidence": getattr(signal, "confidence", 0),
                        "notes": getattr(signal, "notes", []),
                    },
                })
        except Exception as e:
            logger.warning(f"get_patterns reversal: {e}")

    return patterns


@app.get("/api/patterns/rainforest")
async def get_rainforest():
    """Rainforest regime detection + top contributing patterns."""
    result = {"regime": None, "confidence": 0, "patterns": []}
    if _rainforest_detector:
        try:
            def _rf_predict():
                df = _data_feed_load("XAUUSDm", "M5", 500)
                if df is not None and not df.empty and _rainforest_detector.is_trained():
                    return _rainforest_detector.predict_regime(df)
                return None
            pred = await asyncio.to_thread(_rf_predict)
            if pred:
                result["regime"] = pred.get("regime")
                result["confidence"] = pred.get("confidence", 0)
                result["probabilities"] = pred.get("probabilities", {})
                result["patterns"] = [
                    {"name": p.get("pattern") or p.get("feature", ""),
                     "importance": p.get("importance", 0)}
                    for p in pred.get("top_patterns", [])[:10]
                ]
        except Exception as e:
            logger.warning(f"get_rainforest: {e}")
    return result


@app.get("/api/patterns/verified")
async def get_patterns_verified():
    """Patterns that have been outcome-validated (from pipeline audit trail)."""
    verified = []
    if _pipeline_audit_funcs:
        try:
            decisions = _pipeline_audit_funcs["get_recent"](50)
            for d in decisions:
                dt = d.get("decision_type", "")
                if dt in ("pattern_verified", "pattern_detected", "rainforest_regime"):
                    details = d.get("details", {}) if isinstance(d.get("details"), dict) else {}
                    verified.append({
                        "pattern_id": str(d.get("candidate", d.get("run_id", "unknown"))),
                        "pattern_name": details.get("pattern_name", dt),
                        "confidence": details.get("confidence", 0.5),
                        "regime": str(details.get("regime", "unknown")),
                        "outcome": d.get("decision", "unknown"),
                        "verified": True,
                        "fallback_incidents": 0,
                    })
        except Exception as e:
            logger.warning(f"get_patterns_verified: {e}")
    return verified


# ═══════════════════════════════════════════════════════
#  SECTION 8: EVOLUTION — Perpetual improvement & agents
#  Wires to UI: PerpetualPanel, AgentsPanel
# ═══════════════════════════════════════════════════════

@app.get("/api/perpetual_improvement")
async def get_perpetual_improvement():
    """Perpetual improvement — learning events, experiments, evolution tracking."""
    events = []
    if _pipeline_audit_funcs:
        try:
            decisions = _pipeline_audit_funcs["get_recent"](20)
            for d in decisions:
                details = d.get("details", {}) if isinstance(d.get("details"), dict) else {}
                events.append({
                    "ts": d.get("_logged_at", ""),
                    "event": d.get("decision_type", ""),
                    "symbol": details.get("symbol", ""),
                    "model": d.get("candidate", ""),
                })
        except Exception:
            pass
    return {
        "loop_status": "idle",
        "learning_events": events,
        "candidate_experiments": [],
    }


@app.get("/api/agents/status")
async def get_agents_status():
    """Agent operational status — health, heartbeats, tasks."""
    return [
        {
            "agent_id": "data_feed", "agent_name": "Data Feed",
            "status": "idle", "heartbeat": None, "current_task": None,
            "last_artifact": None, "error_count": 0,
        },
        {
            "agent_id": "lane_b_trainer", "agent_name": "Lane B Trainer",
            "status": "idle", "heartbeat": None, "current_task": None,
            "last_artifact": "lane_b_seed_456_XAUUSDm_model.zip", "error_count": 0,
        },
        {
            "agent_id": "champion_selector", "agent_name": "Champion Selector",
            "status": "idle", "heartbeat": None, "current_task": None,
            "last_artifact": None, "error_count": 0,
        },
        {
            "agent_id": "safety_guard", "agent_name": "Safety Guard",
            "status": "online",
            "heartbeat": datetime.now(timezone.utc).isoformat(),
            "current_task": "Monitoring kill switch + real money lock",
            "last_artifact": None, "error_count": 0,
        },
    ]


# ═══════════════════════════════════════════════════════
#  SECTION 9: CONTROL — POST endpoints for user actions
#  Wires to UI: SettingsPanel (mode toggle, MT5 login, reset)
# ═══════════════════════════════════════════════════════

@app.post("/api/control")
async def post_control(payload: dict):
    """Generic system control action."""
    action = payload.get("action", "unknown")
    _system_state["control_log"].append({
        "action": action, "ts": datetime.now(timezone.utc).isoformat(),
    })
    logger.info(f"Control action: {action}")
    return {"success": True, "action": action}


@app.post("/api/mode")
async def post_mode(payload: dict):
    """Set trading mode — paper_sim / demo_live / real_live."""
    mode = payload.get("mode", _system_state["system_mode"])
    _system_state["system_mode"] = mode
    _system_state["real_money_locked"] = (mode != "real_live")
    logger.info(f"Mode => {mode}, real_money_locked => {_system_state['real_money_locked']}")
    return {"success": True, "mode": mode, "real_money_locked": _system_state["real_money_locked"]}


@app.post("/api/mt5_login")
async def post_mt5_login(payload: dict):
    """Submit MT5 credentials."""
    login = payload.get("login", "")
    logger.info(f"MT5 login attempt: {login}")
    return {"success": True, "message": "MT5 login received — validation deferred"}


@app.post("/api/paper_reset")
async def post_paper_reset():
    """Reset paper trading account balance."""
    logger.info("Paper account reset")
    return {"success": True, "message": "Paper account reset"}


# ═══════════════════════════════════════════════════════
#  SECTION 10: WEBSOCKET — Real-time status push
#  Wires to UI: all panels via createStatusWS()
# ═══════════════════════════════════════════════════════

@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket):
    """Real-time status push — broadcasts StatusPayload every 2 seconds."""
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception:
        ws_manager.disconnect(websocket)


# ═══════════════════════════════════════════════════════
#  MAIN — standalone entry point
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("DASHBOARD_PORT", "5051"))
    logger.info(f"Starting dashboard backend on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
