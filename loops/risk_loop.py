"""Agent 6: Risk Loop — exposure, drawdown, kill switch."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.equity_tracker import record_snapshot
from core.risk_manager import RiskManager
from core.utils import load_config, read_json_state, setup_logger, write_json_state


def run() -> dict:
    """Evaluate portfolio risk and update kill switch."""
    config = load_config()
    logger = setup_logger("risk_loop", "risk_loop.log")
    logger.info("Starting risk loop")

    positions_data = read_json_state("paper_positions.json", default={"positions": []})
    orders_data = read_json_state("paper_orders.json", default={"orders": [], "balance": {}})
    trades_data = read_json_state("paper_trades.json", default={"trades": []})
    features = read_json_state("features.json", default={"symbols": {}})
    kill_existing = read_json_state("kill_switch.json", default={"kill_switch": config["risk"].get("kill_switch", False)})

    positions = positions_data.get("positions", [])
    orders = orders_data.get("orders", [])
    balance = dict(orders_data.get("balance", {}))
    trades = trades_data.get("trades", [])

    if config.get("execution", {}).get("mode") == "mt5":
        baseline = read_json_state("mt5_baseline.json", default={})
        account = read_json_state("account.json", default={})
        if baseline.get("starting_cash"):
            balance["starting_cash"] = float(baseline["starting_cash"])
        if account.get("equity") is not None:
            balance["equity"] = float(account["equity"])
        if account.get("balance") is not None:
            balance["cash"] = float(account["balance"])

    manager = RiskManager(config, logger)
    result = manager.evaluate(positions, orders, balance, trades, features, kill_existing)

    write_json_state("risk_state.json", result["risk_state"])
    write_json_state("kill_switch.json", result["kill_switch"])
    record_snapshot(
        result["risk_state"].get("equity", balance.get("equity", 0)),
        result["risk_state"].get("cash", balance.get("cash")),
        source="risk_loop",
        extra={
            "drawdown": result["risk_state"].get("drawdown"),
            "open_positions": result["risk_state"].get("open_positions"),
            "exposure_used_pct": result["risk_state"].get("exposure_used_pct"),
        },
    )
    logger.info("Risk state saved — kill_switch=%s", result["kill_switch"]["kill_switch"])
    return result


if __name__ == "__main__":
    run()