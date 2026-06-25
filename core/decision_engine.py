"""Decision Engine — confidence tree + evidence-first trade decisions."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from core.consensus_gates import ConsensusGates
from core.evidence_engine import EvidenceEngine
from core.explain_report import build_explain_report
from core.setup_classifier import SetupClassifier
from core.strategy_ranker import StrategyRanker
from core.utils import utc_now_iso
from core.trade_limits import max_candidates_per_run
from core.weight_defaults import SUBSYSTEM_WEIGHTS


class DecisionEngine:
    """Combine subsystem votes into actionable candidate signals."""

    SUBSYSTEM_WEIGHTS = SUBSYSTEM_WEIGHTS

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.config = config
        self.logger = logger or logging.getLogger("decision_engine")
        self.evidence_engine = EvidenceEngine(config, logger)
        self.setup_classifier = SetupClassifier(config, logger)
        self.consensus = ConsensusGates(config, logger)
        self.ranker = StrategyRanker(config, logger)
        from core.adaptive_weights import load_weights

        self._weights = load_weights(config)

    def generate_candidates(
        self,
        features_data: dict[str, Any],
        context_data: dict[str, Any],
        edge_scores: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        edge_scores = edge_scores or {}
        evidence_data = self.evidence_engine.compute_all(features_data, context_data)
        candidates: list[dict[str, Any]] = []
        max_candidates = max_candidates_per_run(self.config)
        min_confidence = int(self.config["signals"].get("min_confidence", 70))

        for symbol, feat in features_data.get("symbols", {}).items():
            ctx = context_data.get("symbols", {}).get(symbol, {})
            ev = evidence_data.get("symbols", {}).get(symbol, {})
            setup = self.setup_classifier.classify(feat, ctx, ev)
            if not setup:
                continue

            allowed, rank_info = self.ranker.allow_setup(setup["setup_type"], symbol, ctx, feat)
            if not allowed:
                self.logger.info(
                    "Strategy ranker blocked %s %s — %s (top: %s)",
                    symbol,
                    setup["setup_type"],
                    rank_info.get("reason"),
                    rank_info.get("top_setup"),
                )
                continue

            votes = self._confidence_tree(feat, ctx, ev, setup)
            final_confidence = self._combine_votes(votes, setup, symbol, edge_scores)
            if final_confidence < min_confidence:
                continue

            signal = self._build_signal(symbol, feat, setup, votes, final_confidence, ctx, ev)
            regime = ctx.get("market_regime", {})
            passed, vetoes = self.consensus.evaluate(signal, votes, edge_scores, regime)
            signal["consensus_vetoes"] = vetoes
            signal["consensus_passed"] = passed
            if not passed:
                self.logger.info(
                    "Consensus blocked %s %s %s — %s",
                    symbol,
                    signal["side"],
                    setup["setup_type"],
                    vetoes,
                )
                continue

            signal["strategy_rank"] = rank_info
            signal["explain"] = build_explain_report(signal, edge_scores)
            candidates.append(signal)

        candidates.sort(key=lambda s: s["confidence"], reverse=True)
        trimmed = candidates if max_candidates is None else candidates[:max_candidates]
        self.logger.info(
            "Decision engine: %d candidates (%d returned)",
            len(candidates),
            len(trimmed),
        )
        return trimmed

    def _confidence_tree(
        self,
        feat: dict[str, Any],
        ctx: dict[str, Any],
        ev: dict[str, Any],
        setup: dict[str, Any],
    ) -> dict[str, int]:
        """Each subsystem votes 0-100."""
        side = setup["side"]
        trend_vote = int(ev["trend"] * 100)
        if side == "BUY" and feat.get("m5_trend") == "bearish":
            trend_vote = int(trend_vote * 0.6)
        if side == "SELL" and feat.get("m5_trend") == "bullish":
            trend_vote = int(trend_vote * 0.6)

        return {
            "trend_engine": trend_vote,
            "structure_engine": int(ev["structure"] * 100),
            "momentum_engine": int(ev["momentum"] * 100),
            "volume_engine": int(ev["volume"] * 100),
            "liquidity_engine": int(ev["liquidity"] * 100),
            "volatility_engine": int(ev["volatility"] * 100),
            "risk_engine": int((1 - ev["risk"]) * 100),
        }

    def _combine_votes(
        self,
        votes: dict[str, int],
        setup: dict[str, Any],
        symbol: str,
        edge_scores: dict[str, Any],
    ) -> int:
        weighted = sum(votes[k] * self._weights[k] for k in votes if k in self._weights)
        setup_boost = int(setup.get("setup_confidence", 0.5) * 10)
        edge = edge_scores.get("setups", {}).get(symbol, {}).get(setup["setup_type"], 50)
        edge_adj = (edge - 50) * 0.15
        final = weighted + setup_boost + edge_adj
        return int(min(99, max(0, round(final))))

    def _build_signal(
        self,
        symbol: str,
        feat: dict[str, Any],
        setup: dict[str, Any],
        votes: dict[str, int],
        confidence: int,
        ctx: dict[str, Any],
        ev: dict[str, Any],
    ) -> dict[str, Any]:
        price = feat["price"]
        atr = feat.get("atr", price * 0.001)
        side = setup["side"]
        risk_dist = max(atr * 1.5, price * 0.001)
        reward_dist = risk_dist * 1.5

        if side == "BUY":
            sl = round(price - risk_dist, 5)
            tp1 = round(price + reward_dist, 5)
            tp2 = round(price + reward_dist * 2, 5)
        else:
            sl = round(price + risk_dist, 5)
            tp1 = round(price - reward_dist, 5)
            tp2 = round(price - reward_dist * 2, 5)

        reasons = self._build_reasons(setup, votes, ctx)
        regime = ctx.get("market_regime", {})

        return {
            "signal_id": str(uuid.uuid4()),
            "symbol": symbol,
            "side": side,
            "setup_type": setup["setup_type"],
            "entry": price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "confidence": confidence,
            "confidence_tree": votes,
            "evidence": ev,
            "market_context": {
                "regime": ctx.get("regime"),
                "phase": ctx.get("phase"),
                "move_type": ctx.get("move_type"),
                "market_intent": ctx.get("market_intent"),
                "session": ctx.get("session"),
                "market_regime": regime,
            },
            "reason": setup["reason"],
            "reasons": reasons,
            "created_at": utc_now_iso(),
        }

    def _build_reasons(self, setup: dict, votes: dict[str, int], ctx: dict) -> list[str]:
        reasons = [setup["reason"]]
        if votes.get("structure_engine", 0) >= 75:
            reasons.append("High structure quality")
        if votes.get("volume_engine", 0) >= 75:
            reasons.append("Strong volume confirmation")
        if votes.get("momentum_engine", 0) >= 65:
            reasons.append("Moderate momentum")
        elif votes.get("momentum_engine", 0) < 50:
            reasons.append("Weak momentum — caution")
        if votes.get("risk_engine", 0) >= 80:
            reasons.append("Low risk environment")
        regime = ctx.get("market_regime", {})
        if regime.get("primary"):
            reasons.append(f"Market regime: {regime['primary'].replace('_', ' ')}")
        elif ctx.get("regime") == "trending":
            reasons.append(f"Trending market — {ctx.get('trend_strength', 'unknown')} strength")
        return reasons