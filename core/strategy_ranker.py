"""Strategy Ranking Engine — pick best setup for current market conditions."""

from __future__ import annotations

import logging
from typing import Any

from core.edge_database import EdgeDatabase
from core.setup_library import SETUP_LIBRARY


class StrategyRanker:
    """Rank strategies by historical edge under today's regime and session."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("strategy_ranker")
        self.edge_db = EdgeDatabase(logger)
        quant = config.get("quant", {})
        self.enabled = bool(quant.get("strategy_ranking_enabled", True))
        self.min_samples = int(quant.get("min_rank_sample_size", 3))
        self.min_win_rate = float(quant.get("min_rank_win_rate", 35))
        self.require_top_rank = bool(quant.get("require_top_ranked_setup", True))

    def rank_for_symbol(
        self,
        symbol: str,
        context: dict[str, Any],
        features: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return setups ranked by edge score for current conditions."""
        feat = features or {}
        regime = context.get("market_regime", {})
        primary = regime.get("primary") or context.get("regime", "unknown")
        session = context.get("session", "unknown")
        volatility = feat.get("volatility_regime", "normal")

        rankings = self.edge_db.rank_setups_for_context(
            symbol,
            primary,
            session,
            volatility,
            min_samples=self.min_samples,
        )

        if not rankings:
            rankings = self._default_rankings(primary)

        for i, r in enumerate(rankings):
            r["rank"] = i + 1
            r["regime"] = primary
            r["session"] = session

        return rankings

    def allow_setup(
        self,
        setup_type: str,
        symbol: str,
        context: dict[str, Any],
        features: dict[str, Any] | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Check if setup is top-ranked (or has insufficient data fallback)."""
        if not self.enabled:
            return True, {"allowed": True, "reason": "ranking_disabled"}

        rankings = self.rank_for_symbol(symbol, context, features)
        if not rankings:
            return True, {"allowed": True, "reason": "no_rankings"}

        top = rankings[0]
        match = next((r for r in rankings if r["setup_type"] == setup_type), None)

        if match and match.get("insufficient_data"):
            return True, {
                "allowed": True,
                "reason": "insufficient_data_fallback",
                "rankings": rankings[:5],
            }

        if not self.require_top_rank:
            if match and match.get("win_rate_pct", 0) >= self.min_win_rate:
                return True, {"allowed": True, "rank": match["rank"], "rankings": rankings[:5]}
            return False, {
                "allowed": False,
                "reason": f"win_rate_below_{self.min_win_rate}",
                "rankings": rankings[:5],
            }

        if setup_type == top["setup_type"]:
            return True, {
                "allowed": True,
                "reason": "top_ranked",
                "score": top["score"],
                "rankings": rankings[:5],
            }

        if top.get("insufficient_data"):
            return True, {"allowed": True, "reason": "top_insufficient_fallback", "rankings": rankings[:5]}

        return False, {
            "allowed": False,
            "reason": "not_top_ranked",
            "top_setup": top["setup_type"],
            "top_score": top["score"],
            "rankings": rankings[:5],
        }

    def _default_rankings(self, primary: str) -> list[dict[str, Any]]:
        """Prioritize setups compatible with regime when no history exists."""
        regime_setup_map = {
            "strong_trend": ["trend_continuation", "pullback", "breakout"],
            "weak_trend": ["pullback", "trend_continuation"],
            "range": ["range_fade", "mean_reversion", "liquidity_sweep"],
            "compression": ["compression_breakout", "breakout"],
            "expansion": ["breakout", "trend_continuation"],
            "volatility_spike": [],
        }
        preferred = regime_setup_map.get(primary, list(SETUP_LIBRARY.keys()))
        return [
            {
                "setup_type": s,
                "score": 50 - i * 5,
                "win_rate_pct": 50,
                "total": 0,
                "insufficient_data": True,
            }
            for i, s in enumerate(preferred[:5])
        ]