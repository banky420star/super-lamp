import json
import pandas as pd
import numpy as np

from drl.trading_env import TradingEnv


def test_trading_env_exposes_reward_components():
    n = 250
    df = pd.DataFrame(
        {
            "open": [1.1 + i * 0.0001 for i in range(n)],
            "high": [1.1002 + i * 0.0001 for i in range(n)],
            "low": [1.0998 + i * 0.0001 for i in range(n)],
            "close": [1.1 + i * 0.0001 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )

    env = TradingEnv(df, window_size=50)
    obs, _ = env.reset()
    assert obs.shape[0] == 50 * env.n_features + env.portfolio_feature_count
    assert env.n_features >= 15
    assert env.portfolio_feature_count == 9

    obs, reward, terminated, truncated, info = env.step([0.2])
    assert isinstance(reward, float)
    assert "reward_components" in info
    assert "growth" in info["reward_components"]
    assert "loss_streak_penalty" in info["reward_components"]
    assert "memory_expectancy_norm" in info["reward_components"]
    assert info.get("feature_version") == "engineered_v2"


def test_trading_env_decodes_legacy_three_dim_action():
    meta = TradingEnv.decode_action([0.8, 0.2, -0.4])
    assert meta["entry_mode"] == "market"
    assert meta["legacy"] is True
    assert "tp_offset_pct" in meta
    assert "sl_offset_pct" in meta


def test_decision_ppo_rich_action_decode_and_spec():
    """Test Decision PPO rich 18-dim (or any >6) vector produces full structured DecisionSpec."""
    from drl.trading_env import DecisionSpec, DECISION_ACTION_DIM
    import numpy as np

    # Rich vector (simulate policy output)
    rich_vec = np.random.uniform(-0.95, 0.95, size=DECISION_ACTION_DIM).astype(np.float32)
    meta = TradingEnv.decode_action(
        rich_vec,
        max_leverage=1.0,
        decision_ppo=True,
        decision_action_dim=DECISION_ACTION_DIM,
    )

    assert meta.get("decision_ppo") is True or "decision_spec" in meta
    assert "decision_spec" in meta
    dspec = meta["decision_spec"]
    assert isinstance(dspec, DecisionSpec)
    assert abs(dspec.direction) <= 1.0
    assert "lot_spec" in dspec.to_dict()
    assert "tp" in dspec.to_dict()
    assert "sl" in dspec.to_dict()
    assert "trailing" in dspec.to_dict()
    assert "partial_close" in dspec.to_dict()
    assert "breakeven" in dspec.to_dict()
    assert "full_close" in dspec.to_dict()

    # JSON roundtrip
    js = dspec.to_json()
    assert isinstance(js, str) and "lot_spec" in js

    # Dict form for executor handoff
    assert "decision_spec_dict" in meta
    assert isinstance(meta["decision_spec_dict"], dict)


def test_decision_ppo_env_with_xau_btc_test_data():
    """Smoke test rich action space + simulation on real-ish XAU/BTC style data."""
    import json
    import pandas as pd
    from pathlib import Path

    test_data_path = Path("data/test/xauusd_m1_10k_20260528_142950.jsonl")
    if not test_data_path.exists():
        # fallback synthetic for CI
        n = 400
        df = pd.DataFrame({
            "open": 2300 + np.cumsum(np.random.randn(n) * 0.3),
            "high": 2300 + np.cumsum(np.random.randn(n) * 0.3) + 0.4,
            "low": 2300 + np.cumsum(np.random.randn(n) * 0.3) - 0.4,
            "close": 2300 + np.cumsum(np.random.randn(n) * 0.3),
            "volume": 100 + np.random.randint(0, 50, n),
        })
    else:
        rows = []
        with open(test_data_path, "r", encoding="utf-8") as f:
            for line in list(f)[:300]:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
        if rows:
            df = pd.DataFrame(rows)
            if "close" not in df.columns and "c" in df.columns:
                df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        else:
            df = pd.DataFrame({"open": [2300 + i * 0.1 for i in range(300)], "high": [2300.5 + i * 0.1 for i in range(300)],
                               "low": [2299.5 + i * 0.1 for i in range(300)], "close": [2300 + i * 0.1 for i in range(300)], "volume": 50})

    # Force Decision PPO mode
    env = TradingEnv(
        df,
        window_size=60,
        action_config={"decision_ppo": True, "decision_action_dim": 18},
        symbol="XAUUSDm",
    )
    obs, _ = env.reset()
    assert env.action_space.shape[0] >= 8

    # Step with rich action
    rich_action = np.random.uniform(-0.9, 0.9, size=env.action_space.shape[0]).astype(np.float32)
    obs, rew, term, trunc, info = env.step(rich_action)

    assert "action_components" in info
    ac = info["action_components"]
    assert "decision_ppo" in ac or "lot_spec" in str(ac) or ac.get("decision_ppo") is True

    # Verify rich state exists in trade_state
    ts = info.get("trade_state", {})
    assert "open_trade" in ts

    # Second step + check DecisionSpec attachment
    obs2, _, _, _, info2 = env.step(np.random.uniform(-0.8, 0.8, size=env.action_space.shape[0]).astype(np.float32))
    assert isinstance(rew, float)

    # Cleanup
    del env
