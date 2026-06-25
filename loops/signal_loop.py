"""Agent 3: Decision Loop — evidence-first candidate signals via Decision Engine."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.decision_engine import DecisionEngine
from core.strategy_ranker import StrategyRanker
from core.utils import (
    fail_safe_missing,
    load_config,
    read_json_state,
    setup_logger,
    utc_now_iso,
    write_json_state,
)


def run() -> dict | None:
    """Create evidence-based candidate signals — no verification or execution."""
    config = load_config()
    logger = setup_logger("signal_loop", "signal_loop.log")
    logger.info("=== Decision Loop starting (evidence-first) ===")

    if fail_safe_missing("features.json", logger):
        return None
    if fail_safe_missing("market_context.json", logger):
        return None

    features = read_json_state("features.json")
    market_ctx = read_json_state("market_context.json")
    context_data = market_ctx.get("market_context", market_ctx)
    edge_scores = read_json_state("edge_scores.json", default={"setups": {}, "setup_stats": {}})

    engine = DecisionEngine(config, logger)
    ranker = StrategyRanker(config, logger)
    candidates = engine.generate_candidates(features, context_data, edge_scores)

    strategy_rankings: dict[str, list] = {}
    for symbol, feat in features.get("symbols", {}).items():
        ctx = context_data.get("symbols", {}).get(symbol, {})
        strategy_rankings[symbol] = ranker.rank_for_symbol(symbol, ctx, feat)

    top_explain = candidates[0].get("explain") if candidates else None
    output = {
        "timestamp": utc_now_iso(),
        "engine": "decision_engine",
        "count": len(candidates),
        "candidates": candidates,
        "top_explain": top_explain,
        "strategy_rankings": strategy_rankings,
    }
    write_json_state("candidate_signals.json", output)
    write_json_state("strategy_rankings.json", {
        "timestamp": utc_now_iso(),
        "rankings": strategy_rankings,
    })
    logger.info("Saved %d candidate signals", len(candidates))
    for c in candidates[:3]:
        logger.info(
            "  %s %s %s conf=%d tree=%s",
            c["symbol"],
            c["side"],
            c["setup_type"],
            c["confidence"],
            c.get("confidence_tree"),
        )
    logger.info("=== Decision Loop complete ===")
    return output


if __name__ == "__main__":
    run()