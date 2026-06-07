import numpy as np
import pandas as pd

from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, build_env_feature_matrix, build_lstm_feature_frame
from drl.trading_env import TradingEnv


def _sample_df(rows: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="5min", tz="UTC")
    close = np.linspace(1.0, 1.1, rows) + np.sin(np.arange(rows) / 9.0) * 0.01
    open_ = close + 0.001
    high = np.maximum(open_, close) + 0.002
    low = np.minimum(open_, close) - 0.002
    volume = np.linspace(100, 500, rows)
    return pd.DataFrame({"time": idx, "open": open_, "high": high, "low": low, "close": close, "volume": volume})


def test_lstm_feature_versions_have_expected_shape():
    engineered, engineered_cols = build_lstm_feature_frame(_sample_df(), feature_version=ENGINEERED_V2)
    ultimate, ultimate_cols = build_lstm_feature_frame(_sample_df(), feature_version=ULTIMATE_150)

    assert list(engineered.columns) == engineered_cols
    assert "close" in engineered_cols
    assert len(engineered_cols) == 17
    assert len(ultimate_cols) >= 150
    assert ultimate.shape[1] == len(ultimate_cols)


def test_env_feature_versions_have_expected_shape():
    engineered = build_env_feature_matrix(_sample_df(), feature_version=ENGINEERED_V2)
    ultimate = build_env_feature_matrix(_sample_df(), feature_version=ULTIMATE_150)

    assert engineered.shape[1] == 40
    assert ultimate.shape[1] >= 150


def test_trading_env_supports_ultimate_feature_contract():
    env = TradingEnv(_sample_df(), feature_version=ULTIMATE_150)
    obs, _ = env.reset()

    assert env.feature_version == ULTIMATE_150
    assert obs.shape[0] == env.window_size * env.n_features + env.portfolio_feature_count
