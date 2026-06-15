"""
ML-based directional signal for trading.
Trains an XGBoost classifier on engineered features to predict next-bar direction,
then pipes the probability into the PPO observation as an additional feature.
"""
from __future__ import annotations

import numpy as np
from loguru import logger

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    xgb = None

try:
    from sklearn.ensemble import RandomForestClassifier
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

ML_SIGNAL_FEATURES = 1
"""Number of ML signal features appended to the observation matrix (1 = probability)."""

# Module-level cached model to avoid re-training across parallel vectorized envs
_ml_model_cache = {"model": None, "n_samples": 0, "n_features": 0}

# Module-level cache for Rainforest detector (avoid joblib.load on every env step)
_rf_cache: dict = {"detector": None, "symbol": ""}


def compute_ml_signal(
    feature_matrix: np.ndarray,
    close_prices: np.ndarray,
) -> np.ndarray:
    """
    Train a tree-based classifier on features to predict next-bar direction.

    Args:
        feature_matrix: (n_timesteps, n_features) feature matrix.
        close_prices: (n_timesteps,) close prices for computing the target.

    Returns:
        (n_timesteps, 1) probability of next bar being UP, in [0, 1].
        Returns zeros if no model is available.
    """
    n = len(feature_matrix)
    if n < 50:
        return np.zeros((n, 1), dtype=np.float32)

    # Check cache: if same data dimensions, reuse model (avoids re-training per parallel env)
    global _ml_model_cache
    if _ml_model_cache["model"] is not None:
        if _ml_model_cache["n_samples"] == n and _ml_model_cache["n_features"] == feature_matrix.shape[1]:
            try:
                proba = _ml_model_cache["model"].predict_proba(feature_matrix)[:, 1]
                return proba.astype(np.float32).reshape(-1, 1)
            except Exception:
                _ml_model_cache["model"] = None

    # Target: next-bar binary direction
    target = np.zeros(n, dtype=np.int32)
    target[:-1] = (close_prices[1:] > close_prices[:-1]).astype(np.int32)

    # Try XGBoost first (fast, handles non-linearity well)
    model = None
    if _HAS_XGB:
        try:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=42,
                verbosity=0,
                n_jobs=1,
            )
            model.fit(feature_matrix, target)
            logger.info(f"ML Signal: XGBoost trained on {n} x {feature_matrix.shape[1]}")
        except Exception as e:
            logger.warning(f"ML Signal: XGBoost failed ({e})")

    # Fallback to sklearn RandomForest
    if model is None and _HAS_SKLEARN:
        try:
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(feature_matrix, target)
            logger.info(f"ML Signal: RandomForest trained on {n} x {feature_matrix.shape[1]}")
        except Exception as e:
            logger.warning(f"ML Signal: RandomForest failed ({e})")

    if model is None:
        logger.warning("ML Signal: No model available, returning zeros")
        return np.zeros((n, 1), dtype=np.float32)

    proba = model.predict_proba(feature_matrix)[:, 1]
    # Cache the trained model for reuse across parallel envs
    _ml_model_cache["model"] = model
    _ml_model_cache["n_samples"] = n
    _ml_model_cache["n_features"] = feature_matrix.shape[1]
    return proba.astype(np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Rainforest ml_signal — derived from RainforestDetector regime probabilities
# ---------------------------------------------------------------------------

RF_ML_SIGNAL_FEATURES = 1
"""Number of Rainforest ML signal features appended to the observation matrix."""


def compute_rainforest_ml_signal(
    symbol: str,
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Compute Rainforest-based ml_signal from OHLCV data.

    Uses the RainforestDetector to produce per-bar directional signals
    in [0, 1] from regime probabilities. Maps bullish regimes
    (bull_trend, breakout_up, reversal_up) to up-probability.

    This signal is complementary to compute_ml_signal() which trains its
    own XGBoost/RF model — the Rainforest version derives direction from
    regime classification probabilities.

    Args:
        symbol: Trading symbol (e.g. "XAUUSDm").
        df: OHLCV DataFrame with columns open, high, low, close, volume.

    Returns:
        (n_timesteps, 1) array of ml_signal values in [0, 1].
        Returns 0.5 (neutral) if detector is unavailable or untrained.
    """
    n = len(df)
    if n < 50:
        return np.full((n, 1), 0.5, dtype=np.float32)

    global _rf_cache

    # Use cached detector if symbol matches (avoids joblib.load on every env step)
    if _rf_cache["detector"] is not None and _rf_cache["symbol"] == symbol:
        try:
            return _rf_cache["detector"].predict_ml_signal(df)
        except Exception:
            pass  # Fall through to reload

    try:
        from Python.rainforest_detector import RainforestDetector

        detector = RainforestDetector()
        safe_sym = symbol.replace("/", "_")
        import os as _os
        model_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "models",
            f"rainforest_{safe_sym}.pkl",
        )
        loaded = detector.load(model_path)
        if not loaded and not detector.is_trained():
            detector.train_from_mt5_data(symbol)

        if not detector.is_trained():
            return np.full((n, 1), 0.5, dtype=np.float32)

        # Cache for subsequent calls
        _rf_cache["detector"] = detector
        _rf_cache["symbol"] = symbol

        return detector.predict_ml_signal(df)

    except Exception:
        return np.full((n, 1), 0.5, dtype=np.float32)
