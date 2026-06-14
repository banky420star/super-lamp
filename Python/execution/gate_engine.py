"""Trade-intent gate engine.

Receives a trade intent, runs pre-flight checks (spread, regime, telemetry,
test health), and returns a structured gate result.

Phase 6: Hard safety execution layer. Defaults to locked/dry-run.
All gates must pass before ANY execution. "No real-money live trading until every gate passes".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from Python.execution.mode_resolver import resolve_mode
from Python.execution.account_verifier import verify_account
from Python.execution.live_gate import live_trading_allowed, demo_trading_allowed
from Python.execution.risk_supervisor import RiskSupervisor

RUNTIME_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "runtime")
KILL_SWITCH_PATH = os.path.join(RUNTIME_DIR, "KILL_SWITCH")

# Default allowed symbols (Phase 6 spec)
DEFAULT_ALLOWED_SYMBOLS = {"XAUUSDm", "EURUSDm", "GBPUSDm", "BTCUSDm", "ETHUSDm"}

def _is_kill_switch_active() -> bool:
    """Hard kill switch: presence of runtime/KILL_SWITCH blocks ALL trading (real or demo)."""
    try:
        return os.path.exists(KILL_SWITCH_PATH)
    except Exception:
        return False

def _check_symbol_allowed(symbol: str, config: dict) -> tuple[bool, str]:
    allowed = set(config.get("risk", {}).get("allowed_symbols", []) or DEFAULT_ALLOWED_SYMBOLS)
    if not allowed:
        allowed = DEFAULT_ALLOWED_SYMBOLS
    if str(symbol) not in allowed:
        return False, f"symbol_not_allowed:{symbol}"
    return True, "ok"

def _check_market_open(symbol: str) -> tuple[bool, str]:
    # Simplified: rely on MT5 or basic weekday check; full impl in data_feed/market_guardian
    return True, "market_check_deferred_to_executor"

def _check_stale_data(data_ts: Any = None, max_age_sec: float = 300.0) -> tuple[bool, str]:
    """Block if data is stale (> max_age_sec old)."""
    if data_ts is None:
        return True, "stale_check_no_ts"
    try:
        import time
        now = time.time()
        age = now - float(data_ts)
        if age > max_age_sec:
            return False, f"data_stale age={age:.0f}s > {max_age_sec}s"
    except Exception:
        pass
    return True, "ok"

def _check_model_exists(model_path: str | None) -> tuple[bool, str]:
    if not model_path:
        return False, "no_model_path"
    try:
        if not os.path.exists(str(model_path)):
            return False, f"model_not_found:{model_path}"
    except Exception:
        return False, "model_path_error"
    return True, "ok"

def _check_confidence(conf: float, min_conf: float = 0.6) -> tuple[bool, str]:
    if float(conf or 0.0) < min_conf:
        return False, f"confidence_too_low {conf} < {min_conf}"
    return True, "ok"


@dataclass
class GateResult:
    gate_passed: bool
    risk_passed: bool
    execution_mode: str
    reason: str
    blocked_by_safety: bool = False
    dry_run: bool = True
    # Phase 9 enriched for decision logging
    details: dict | None = None


class GateEngine:
    """Central gate for every trade intent.

    Usage:
        engine = GateEngine(config, risk_supervisor)
        result = engine.check_intent(intent)
    """

    def __init__(
        self,
        config: dict | None = None,
        risk_supervisor: RiskSupervisor | None = None,
    ):
        self.config = config or {}
        self.risk = risk_supervisor
        self._mode = resolve_mode(self.config)

    def check_intent(
        self,
        intent: dict[str, Any],
        account_state: dict[str, Any] | None = None,
        validation_state: dict[str, Any] | None = None,
        test_state: dict[str, Any] | None = None,
    ) -> GateResult:
        """Evaluate a single trade intent against all gates.

        Intent fields expected:
          symbol, side, size, spread_bps, regime, target_exposure
        """
        symbol = intent.get("symbol", "")
        spread_bps = float(intent.get("spread_bps", 0.0) or 0.0)
        regime = str(intent.get("regime", "")).lower()
        target_exposure = float(intent.get("target_exposure", 0.0) or 0.0)
        confidence = float(intent.get("confidence", intent.get("agi_confidence", 0.0)) or 0.0)
        model_path = intent.get("model_path") or intent.get("model_candidate_dir")
        data_ts = intent.get("data_ts") or intent.get("last_bar_time")

        # === PHASE 6 HARD GATES (must pass before any execution; defaults: locked + dry-run) ===
        if _is_kill_switch_active():
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason="kill_switch_active (runtime/KILL_SWITCH present)",
                blocked_by_safety=True,
                dry_run=True,
                details={"kill_switch": KILL_SWITCH_PATH},
            )

        # real_money_locked default: only real_live is real money, everything else locked/dry
        dry_run = self._mode not in ("real_live", "demo_live")
        if self._mode == "real_live_locked":
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason="real_money_locked (CHAIN_GAMBLER_ALLOW_LIVE!=1 or mode=real_live_locked)",
                blocked_by_safety=True,
                dry_run=True,
                details={"mode": self._mode},
            )

        # Symbol allowed list gate
        sym_ok, sym_reason = _check_symbol_allowed(symbol, self.config)
        if not sym_ok:
            return GateResult(
                gate_passed=False, risk_passed=False, execution_mode=self._mode,
                reason=sym_reason, blocked_by_safety=True, dry_run=dry_run,
                details={"allowed_symbols": list(DEFAULT_ALLOWED_SYMBOLS)},
            )

        # Stale data block
        stale_ok, stale_reason = _check_stale_data(data_ts)
        if not stale_ok:
            return GateResult(
                gate_passed=False, risk_passed=False, execution_mode=self._mode,
                reason=stale_reason, blocked_by_safety=True, dry_run=dry_run,
            )

        # Market closed (soft for now, defer full to executor but gate if known closed)
        mkt_ok, mkt_reason = _check_market_open(symbol)
        if not mkt_ok:
            return GateResult(
                gate_passed=False, risk_passed=False, execution_mode=self._mode,
                reason=mkt_reason, blocked_by_safety=True, dry_run=dry_run,
            )

        # Model exists (if path supplied in intent)
        if model_path:
            mod_ok, mod_reason = _check_model_exists(model_path)
            if not mod_ok:
                return GateResult(
                    gate_passed=False, risk_passed=False, execution_mode=self._mode,
                    reason=mod_reason, blocked_by_safety=True, dry_run=dry_run,
                )

        # Confidence gate (hard min, default 0.6; config can override)
        min_conf = float(self.config.get("risk", {}).get("min_confidence", 0.60))
        conf_ok, conf_reason = _check_confidence(confidence, min_conf)
        if not conf_ok:
            return GateResult(
                gate_passed=False, risk_passed=False, execution_mode=self._mode,
                reason=conf_reason, blocked_by_safety=True, dry_run=dry_run,
            )

        # 1. Spread gate
        max_spread_bps = float(
            self.config.get("risk", {}).get("max_spread_bps", 50.0)
        )
        if spread_bps > max_spread_bps:
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason=f"spread_too_high ({spread_bps:.1f} > {max_spread_bps})",
            )

        # 2. Regime gate — block chaos / spread danger
        blocked_regimes = {"chaos_spike", "spread_danger", "black_swan", "halt"}
        if regime in blocked_regimes:
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason=f"regime_blocked ({regime})",
            )

        # 3. Account telemetry gate
        account = account_state or {}
        if not account.get("telemetry_valid", True):
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason="account_telemetry_invalid",
            )

        # 4. Test-state gate
        tests = test_state or {}
        if tests.get("tests_clean") is False:
            return GateResult(
                gate_passed=False,
                risk_passed=False,
                execution_mode=self._mode,
                reason="tests_failing",
            )

        # 5. Risk-supervisor gate (drawdown, daily loss, trade count, etc.)
        risk_ok = True
        risk_reason = "ok"
        if self.risk is not None:
            risk_ok = self.risk.can_trade(symbol)
            if not risk_ok:
                risk_reason = getattr(self.risk, "_halt_reason", "risk_blocked")

        if not risk_ok:
            return GateResult(
                gate_passed=True,  # structural gates passed, risk blocked
                risk_passed=False,
                execution_mode=self._mode,
                reason=risk_reason,
            )

        # 6. Mode-specific gates
        if self._mode == "real_live":
            allowed, reason = live_trading_allowed(
                self.config,
                validation_state or {},
                account,
                tests,
            )
            if not allowed:
                return GateResult(
                    gate_passed=False,
                    risk_passed=False,
                    execution_mode=self._mode,
                    reason=reason,
                )

        elif self._mode == "demo_live":
            risk_state = {
                "halt": self.risk.halt if self.risk else False,
                "daily_pnl": getattr(self.risk, "realized_pnl_today", 0.0) if self.risk else 0.0,
                "max_daily_loss": getattr(self.risk, "max_daily_loss", 1000.0) if self.risk else 1000.0,
                "open_positions": int(intent.get("open_positions", 0)),
                "max_open_positions": getattr(self.risk, "max_open_positions", 6) if self.risk else 6,
            }
            allowed, reason = demo_trading_allowed(
                self.config,
                account,
                risk_state,
            )
            if not allowed:
                return GateResult(
                    gate_passed=False,
                    risk_passed=False,
                    execution_mode=self._mode,
                    reason=reason,
                )

        # All gates cleared
        return GateResult(
            gate_passed=True,
            risk_passed=True,
            execution_mode=self._mode,
            reason="ok",
        )
