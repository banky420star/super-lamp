"""
Regime classifier for trending vs ranging markets.

Provides:
- ADX, ATR, Bollinger Band calculations (pure NumPy, no TA library needed)
- classify_regime(): returns regime (0=ranging, 1=trending) and score for each bar
- RegimeAwareEnv: extends TamedOHLCVEnv with regime features in observation
"""
import numpy as np
import pandas as pd


# ── Indicators ──

def _true_range(high, low, close_prev):
    return max(high - low, abs(high - close_prev), abs(low - close_prev))


def atr(high, low, close, period=14):
    """Average True Range (Wilder smoothing)."""
    tr = np.zeros_like(close)
    tr[0] = high[0] - low[0]
    for i in range(1, len(close)):
        tr[i] = _true_range(high[i], low[i], close[i - 1])
    atr_vals = np.zeros_like(close)
    atr_vals[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, len(close)):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period
    return atr_vals


def _directional_movement(high, low):
    """Calculate +DM and -DM arrays."""
    n = len(high)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up_move = high[i] - high[i - 1]
        down_move = low[i - 1] - low[i]
        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        elif down_move > up_move and down_move > 0:
            minus_dm[i] = down_move
    return plus_dm, minus_dm


def adx(high, low, close, period=14):
    """
    Average Directional Index.

    Returns
    -------
    adx_vals  : (n,) ADX values
    plus_di   : (n,) +DI values
    minus_di  : (n,) -DI values
    """
    n = len(high)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = _true_range(high[i], low[i], close[i - 1])

    plus_dm, minus_dm = _directional_movement(high, low)

    # Wilder smoothing
    smoothed_tr = np.zeros(n)
    smoothed_plus = np.zeros(n)
    smoothed_minus = np.zeros(n)
    smoothed_tr[period] = np.sum(tr[1:period + 1])
    smoothed_plus[period] = np.sum(plus_dm[1:period + 1])
    smoothed_minus[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        smoothed_tr[i] = smoothed_tr[i - 1] - smoothed_tr[i - 1] / period + tr[i]
        smoothed_plus[i] = smoothed_plus[i - 1] - smoothed_plus[i - 1] / period + plus_dm[i]
        smoothed_minus[i] = smoothed_minus[i - 1] - smoothed_minus[i - 1] / period + minus_dm[i]

    plus_di = np.where(smoothed_tr > 0, smoothed_plus / smoothed_tr * 100, 0.0)
    minus_di = np.where(smoothed_tr > 0, smoothed_minus / smoothed_tr * 100, 0.0)
    dx = np.where((plus_di + minus_di) > 0,
                  np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0.0)

    # ADX = smoothed DX
    adx_vals = np.zeros(n)
    adx_vals[period * 2] = np.mean(dx[:period * 2]) if len(dx) > period * 2 else 0.0
    for i in range(period * 2 + 1, n):
        adx_vals[i] = (adx_vals[i - 1] * (period - 1) + dx[i]) / period

    return adx_vals, plus_di, minus_di


def bollinger_bands(close, period=20, num_std=2):
    """Bollinger Bands. Returns upper, lower, %b, bandwidth."""
    s = pd.Series(close)
    sma = s.rolling(period).mean().values
    std = s.rolling(period).std(ddof=0).values
    upper = sma + num_std * std
    lower = sma - num_std * std
    spread = upper - lower
    bb_pct = np.where(spread > 0, (close - lower) / spread, 0.5)
    bb_width = np.where(np.abs(sma) > 1e-10, spread / np.abs(sma), 0.0)
    return upper, lower, bb_pct, bb_width


def classify_regime(df, adx_period=14, bb_period=20,
                    trend_threshold=25, ranging_threshold=20):
    """
    Classify each bar as ranging (0) or trending (1).

    Trending = ADX >= 25 (strong trend in either direction)
    Ranging = ADX <= 20 (weak or no trend)
    20-25 = transition zone: leans based on +DI vs -DI

    Returns
    -------
    regimes       : (n,) int32 — 0=ranging, 1=trending
    regime_scores : (n,) float32 — continuous from -1 (strong downtrend)
                    to +1 (strong uptrend), near 0 = ranging
    """
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    adx_vals, plus_di, minus_di = adx(high, low, close, adx_period)
    _, _, bb_pct, _ = bollinger_bands(close, bb_period)

    regimes = np.zeros(n, dtype=np.int32)
    scores = np.zeros(n, dtype=np.float32)

    for i in range(n):
        if np.isnan(adx_vals[i]) or np.isnan(plus_di[i]):
            regimes[i] = 0
            scores[i] = 0.0
            continue

        if adx_vals[i] >= trend_threshold:
            regimes[i] = 1  # trending
        elif adx_vals[i] <= ranging_threshold:
            regimes[i] = 0  # ranging
        else:
            # Transition zone: lean based on +DI > -DI
            regimes[i] = 1 if plus_di[i] > minus_di[i] else 0

        # Continuous score: (+DI - -DI) / 100 * min(ADX/25, 1)
        di_diff = (plus_di[i] - minus_di[i]) / 100.0  # -1 to +1
        adx_factor = min(adx_vals[i] / 25.0, 1.0)
        scores[i] = di_diff * adx_factor

    return regimes, scores


def compute_regime_features(df, adx_period=14, bb_period=20):
    """
    Build regime feature array: one float per bar.

    Feature = regime_score (continuous -1 to +1, 0 = ranging)
    """
    _, scores = classify_regime(df, adx_period, bb_period)
    return scores.reshape(-1, 1)


def get_regime_summary(regimes, labels=None):
    """Print a summary of regime distribution."""
    ranging = np.sum(regimes == 0)
    trending = np.sum(regimes == 1)
    total = len(regimes)
    print(f"  Regime distribution:")
    print(f"    Ranging : {ranging:>7d} bars ({ranging / total * 100:5.1f}%)")
    print(f"    Trending: {trending:>7d} bars ({trending / total * 100:5.1f}%)")
    return {"ranging": int(ranging), "trending": int(trending)}
