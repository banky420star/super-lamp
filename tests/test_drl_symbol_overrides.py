from drl.trading_env import TradingEnv
from training.train_drl import _resolve_symbol_training_options


def test_btc_symbol_training_options_override_global_reward_and_action():
    cfg = {
        "drl": {
            "feature_version": "ultimate_150",
            "reward": {
                "version": "base",
                "weights": {
                    "growth": 8.0,
                    "churn_penalty": 0.5,
                },
            },
            "symbol_overrides": {
                "BTCUSDm": {
                    "reward": {
                        "version": "btc_only",
                        "weights": {
                            "churn_penalty": 0.7,
                            "neutral_collapse_penalty": 0.9,
                        },
                    },
                    "action": {
                        "min_target_abs": 0.015,
                    },
                }
            },
        }
    }

    reward_cfg, reward_weights, action_cfg, feature_version, reward_scale, penalty_scale = _resolve_symbol_training_options(
        cfg,
        ["BTCUSDm"],
        default_feature_version="ultimate_150",
    )
    # NEW scales default to 1.0 (hardened); test focuses on weights/overrides
    assert reward_scale == 1.0
    assert penalty_scale == 1.0

    assert reward_cfg["version"] == "btc_only"
    assert reward_weights["growth"] == 8.0
    assert reward_weights["churn_penalty"] == 0.7
    assert reward_weights["neutral_collapse_penalty"] == 0.9
    assert action_cfg["min_target_abs"] == 0.015
    assert feature_version == "engineered_v2"


def test_decode_action_respects_lower_symbol_thresholds():
    action = [0.02, 0.2, 0.0, 0.0, 0.0, 0.0]

    default_meta = TradingEnv.decode_action(action, max_leverage=1.0)
    btc_meta = TradingEnv.decode_action(
        action,
        max_leverage=1.0,
        min_direction_abs=0.015,
        min_size_abs=0.015,
        min_target_abs=0.01,
    )

    assert default_meta["target"] != 0.0  # default threshold 0.005 allows target=0.02 through
    assert btc_meta["target"] != 0.0
