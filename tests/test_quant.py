"""Tests for edge database, adaptive weights, strategy ranker, and research helpers."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.adaptive_weights import AdaptiveWeightOptimizer, load_weights
from core.edge_database import EdgeDatabase
from core.strategy_ranker import StrategyRanker
from core.trade_enrichment import enrich_trade
from loops.research_loop import _find_patterns, _replay_score


@pytest.fixture
def config():
    from core.utils import load_config
    return copy.deepcopy(load_config())


def _sample_trade(trade_id: str, result: str, tree: dict) -> dict:
    return {
        "trade_id": trade_id,
        "signal_id": f"sig-{trade_id}",
        "symbol": "XAUUSDm",
        "side": "BUY",
        "setup_type": "trend_continuation",
        "entry": 2000.0,
        "exit": 2010.0 if result == "win" else 1990.0,
        "sl": 1990.0,
        "tp1": 2015.0,
        "pnl": 10.0 if result == "win" else -10.0,
        "result": result,
        "confidence": 75,
        "confidence_tree": tree,
        "market_context": {
            "session": "London",
            "market_regime": {"primary": "strong_trend"},
        },
        "closed_at": "2026-06-25T10:00:00Z",
    }


def test_edge_database_ingest_and_query(tmp_path, monkeypatch):
    from core import utils

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(utils, "STATE_DIR", state_dir)

    db = EdgeDatabase()
    features = {"symbols": {"XAUUSDm": {"volatility_regime": "high", "spread_points": 12}}}
    context = {"symbols": {"XAUUSDm": {"session": "London", "regime": "trending"}}}

    trade = _sample_trade("t1", "win", {"trend_engine": 80, "structure_engine": 70})
    rec = db.ingest_trade(trade, features=features, context=context, source="test")
    assert rec is not None
    assert rec["setup_type"] == "trend_continuation"
    assert rec["session"] == "London"
    assert rec["volatility"] == "high"

    dup = db.ingest_trade(trade, features=features, context=context)
    assert dup is None

    stats = db.query_win_rate(setup_type="trend_continuation", session="London")
    assert stats["total"] == 1
    assert stats["win_rate_pct"] == 100.0


def test_edge_database_rank_setups(tmp_path, monkeypatch):
    from core import utils

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(utils, "STATE_DIR", state_dir)

    db = EdgeDatabase()
    ctx = {"symbols": {"XAUUSDm": {"session": "London"}}}
    tree = {"trend_engine": 70, "structure_engine": 70}

    for i in range(4):
        db.ingest_trade(
            _sample_trade(f"w{i}", "win", tree),
            context=ctx,
            source="test",
        )
    for i in range(2):
        trade = _sample_trade(f"l{i}", "loss", tree)
        trade["setup_type"] = "range_fade"
        db.ingest_trade(trade, context=ctx, source="test")

    rankings = db.rank_setups_for_context("XAUUSDm", "strong_trend", "London", min_samples=3)
    assert rankings
    assert rankings[0]["setup_type"] == "trend_continuation"


def test_adaptive_weights_optimize(tmp_path, monkeypatch, config):
    from core import utils

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(utils, "STATE_DIR", state_dir)

    db = EdgeDatabase()
    for i in range(10):
        db.ingest_trade(
            _sample_trade(
                f"t{i}",
                "win" if i % 2 == 0 else "loss",
                {
                    "trend_engine": 85 if i % 2 == 0 else 40,
                    "structure_engine": 80 if i % 2 == 0 else 45,
                    "momentum_engine": 60,
                    "volume_engine": 55,
                    "liquidity_engine": 50,
                    "volatility_engine": 50,
                    "risk_engine": 70,
                },
            ),
            source="test",
        )

    opt = AdaptiveWeightOptimizer(config)
    result = opt.optimize(min_trades=8)
    assert result["status"] == "candidate"
    assert "trend_engine" in result["weights"]
    assert abs(sum(result["weights"].values()) - 1.0) < 0.02


def test_load_weights_defaults(config):
    weights = load_weights(config)
    assert "trend_engine" in weights
    assert sum(weights.values()) > 0.9


def test_strategy_ranker_allow_setup(config):
    ranker = StrategyRanker(config)
    ctx = {"session": "London", "market_regime": {"primary": "strong_trend"}}
    allowed, info = ranker.allow_setup("trend_continuation", "XAUUSDm", ctx, {})
    assert allowed is True
    assert info["allowed"] is True


def test_trade_enrichment():
    trade = {"trade_id": "t1", "signal_id": "sig-1", "symbol": "XAUUSDm", "result": "win"}
    index = {
        "sig-1": {
            "signal_id": "sig-1",
            "confidence": 82,
            "confidence_tree": {"trend_engine": 90},
            "setup_type": "breakout",
            "market_context": {"session": "Asia"},
        }
    }
    enriched = enrich_trade(trade, index)
    assert enriched["confidence"] == 82
    assert enriched["setup_type"] == "breakout"
    assert enriched["signal_meta"]["confidence_tree"]["trend_engine"] == 90


def test_replay_score_and_patterns(tmp_path, monkeypatch):
    from core import utils

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    monkeypatch.setattr(utils, "STATE_DIR", state_dir)

    db = EdgeDatabase()
    ctx = {"symbols": {"XAUUSDm": {"session": "London"}}}
    tree = {"trend_engine": 70, "structure_engine": 70}
    for i in range(6):
        db.ingest_trade(_sample_trade(f"p{i}", "win", tree), context=ctx, source="test")

    patterns = _find_patterns(db)
    assert patterns
    assert patterns[0]["type"] == "high_edge"
    assert _replay_score({"pnl_total": 50, "win_rate_pct": 60, "trades_closed": 10}) > 0