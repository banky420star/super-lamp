"""Market Context Loop — what is the market trying to do?"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evidence_engine import EvidenceEngine
from core.market_context import MarketContextEngine
from core.market_regime import MarketRegimeEngine
from core.setup_library import list_setups
from core.utils import fail_safe_missing, load_config, read_json_state, setup_logger, write_json_state


def run() -> dict | None:
    config = load_config()
    logger = setup_logger("market_context_loop", "market_context_loop.log")
    logger.info("=== Market Context Loop starting ===")

    if fail_safe_missing("features.json", logger):
        return None

    features = read_json_state("features.json")
    candles = read_json_state("latest_candles.json", default={})

    ctx_engine = MarketContextEngine(logger)
    context = ctx_engine.analyze_all(features, candles)

    regime_engine = MarketRegimeEngine(logger)
    regimes = regime_engine.classify_all(features, context)
    for symbol, ctx in context.get("symbols", {}).items():
        ctx["market_regime"] = regimes.get("symbols", {}).get(symbol, {})

    evidence_engine = EvidenceEngine(config, logger)
    evidence = evidence_engine.compute_all(features, context)

    output = {
        "timestamp": context["timestamp"],
        "market_context": context,
        "market_regime": regimes,
        "evidence": evidence,
        "setup_library": list_setups(),
    }
    write_json_state("market_context.json", output)
    logger.info("Saved market_context.json for %d symbols", len(context.get("symbols", {})))
    logger.info("=== Market Context Loop complete ===")
    return output


if __name__ == "__main__":
    run()