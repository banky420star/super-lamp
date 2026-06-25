"""Replay Engine — backtest agent logic bar-by-bar on Parquet history."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pandas as pd

from core.decision_engine import DecisionEngine
from core.edge_database import EdgeDatabase
from core.feature_engine import FeatureEngine
from core.history_manager import HistoryManager
from core.market_context import MarketContextEngine
from core.market_regime import MarketRegimeEngine
from core.memory_engine import MemoryEngine
from core.paper_broker import PaperBroker
from core.trade_enrichment import enrich_trades
from core.utils import utc_now_iso, write_json_state
from core.verifier import Verifier


class ReplayEngine:
    """Walk historical candles and simulate full agent pipeline in paper mode."""

    def __init__(self, config: dict[str, Any], logger: logging.Logger | None = None):
        self.config = copy.deepcopy(config)
        self.logger = logger or logging.getLogger("replay_engine")
        self.replay_cfg = self.config.get("replay", {})
        self.config["execution"]["mode"] = "paper"
        self.history = HistoryManager(self.config, logger=self.logger)

    def run(
        self,
        symbol: str | None = None,
        max_bars: int | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        symbol = symbol or self.replay_cfg.get("symbol") or self.config["mt5"]["symbols"][0]
        max_bars = max_bars or int(self.replay_cfg.get("max_bars", 500))
        step = step or int(self.replay_cfg.get("bars_per_step", 5))
        min_bars = int(self.config.get("features", {}).get("min_bars_required", 20))

        m5_df = self.history.load(symbol, "M5")
        m15_df = self.history.load(symbol, "M15")
        if m5_df is None or m5_df.empty:
            raise RuntimeError(f"No M5 history for {symbol}")

        m5_df = m5_df.sort_values("time").reset_index(drop=True)
        if m15_df is not None and not m15_df.empty:
            m15_df = m15_df.sort_values("time").reset_index(drop=True)

        start_idx = max(min_bars, len(m5_df) - max_bars)
        feature_engine = FeatureEngine(self.config, history_manager=None, logger=self.logger)
        ctx_engine = MarketContextEngine(self.logger)
        regime_engine = MarketRegimeEngine(self.logger)
        decision = DecisionEngine(self.config, self.logger)
        verifier = Verifier(self.config, self.logger)
        broker = PaperBroker(self.config, self.logger)
        memory = MemoryEngine(self.logger)
        edge_db = EdgeDatabase(self.logger)
        ingest_edge = bool(self.config.get("quant", {}).get("replay_ingest_edge_db", True))

        orders: list[dict] = []
        positions: list[dict] = []
        trades: list[dict] = []
        approved_history: list[dict] = []
        balance = {
            "cash": float(self.config["execution"].get("starting_cash", 1000)),
            "equity": float(self.config["execution"].get("starting_cash", 1000)),
            "starting_cash": float(self.config["execution"].get("starting_cash", 1000)),
        }
        edge_scores: dict[str, Any] = {"setups": {}, "setup_stats": {}}
        memory_state: dict[str, Any] = {"records": [], "adjustments": []}
        signals_generated = 0
        signals_approved = 0

        for i in range(start_idx, len(m5_df), step):
            m5_slice = m5_df.iloc[: i + 1]
            bar_time = m5_slice["time"].iloc[-1]
            m15_slice = self._m15_up_to(m15_df, bar_time) if m15_df is not None else pd.DataFrame()

            if len(m5_slice) < min_bars:
                continue
            if m15_slice is not None and len(m15_slice) < min_bars:
                continue

            candles = {
                "source": "replay",
                "symbols": {
                    symbol: {
                        "M5": m5_slice.to_dict("records"),
                        "M15": m15_slice.to_dict("records") if len(m15_slice) else [],
                    }
                },
            }
            features = feature_engine.compute_all(candles)
            if symbol not in features.get("symbols", {}):
                continue

            context = ctx_engine.analyze_all(features, candles)
            regimes = regime_engine.classify_all(features, context)
            for sym, ctx in context.get("symbols", {}).items():
                ctx["market_regime"] = regimes.get("symbols", {}).get(sym, {})

            price = features["symbols"][symbol]["price"]
            prices = {symbol: price}
            candidates = decision.generate_candidates(features, context, edge_scores)
            signals_generated += len(candidates)

            approved, _rejected = verifier.verify_batch(
                candidates,
                features,
                active_signals=positions,
                spread_data={symbol: 0.0},
                equity=balance.get("equity", balance["cash"]),
            )
            signals_approved += len(approved)

            if approved:
                approved_history.extend(approved)
                result = broker.process_approved_signals(
                    approved, prices, orders, positions, trades, balance,
                )
                prior_trade_count = len(trades)
                orders = result["orders"]
                positions = result["positions"]
                trades = result["trades"]
                balance = result["balance"]

                if ingest_edge and len(trades) > prior_trade_count:
                    new_closed = trades[prior_trade_count:]
                    enriched = enrich_trades(
                        new_closed,
                        approved_data={"approved": approved_history},
                        orders=orders,
                    )
                    edge_db.ingest_batch(
                        enriched,
                        features=features,
                        context=context,
                        source="replay",
                    )

            mem_out = memory.process(trades, features, context, memory_state, edge_scores)
            memory_state = mem_out["memory"]
            edge_scores = mem_out["edge_scores"]

        wins = sum(1 for t in trades if t.get("result") == "win")
        losses = len(trades) - wins
        pnl = sum(float(t.get("pnl", 0)) for t in trades)

        output = {
            "timestamp": utc_now_iso(),
            "symbol": symbol,
            "bars_replayed": len(range(start_idx, len(m5_df), step)),
            "signals_generated": signals_generated,
            "signals_approved": signals_approved,
            "trades_closed": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(wins / len(trades) * 100, 1) if trades else 0,
            "pnl_total": round(pnl, 2),
            "final_equity": round(balance.get("equity", 0), 2),
            "starting_cash": balance.get("starting_cash"),
            "trades": trades[-50:],
            "setup_stats": edge_scores.get("setup_stats", {}),
        }
        write_json_state("replay_results.json", output)
        self.logger.info(
            "Replay %s: %d bars, %d trades, PnL=%.2f, win rate=%.1f%%",
            symbol,
            output["bars_replayed"],
            len(trades),
            pnl,
            output["win_rate_pct"],
        )
        return output

    @staticmethod
    def _m15_up_to(m15_df: pd.DataFrame | None, bar_time: Any) -> pd.DataFrame:
        if m15_df is None or m15_df.empty:
            return pd.DataFrame()
        return m15_df[m15_df["time"] <= bar_time]