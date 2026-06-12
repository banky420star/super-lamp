"""
Lane C Live Trading Bot — Meta-Controller with Regime-Weighted PPO Models.

Connects to MT5, loads trained range-weighted and trend-weighted models,
runs H1 ADX regime detection in real-time, and executes trades via meta-controller.

Usage
-----
    # First, train and save models:
    python training/run_lane_c_mtf_regime.py --steps 50000 --seed 42

    # Then run live trading:
    python training/live_trade_lane_c.py --symbol XAUUSDm --risk 0.02

    # Dry-run (no real orders, just log decisions):
    python training/live_trade_lane_c.py --dry-run

CLI args:
    --symbol       Trading symbol (default: XAUUSDm)
    --risk         Risk per trade as fraction of equity (default: 0.02 = 2%)
    --range-model  Path to range-weighted model (default: runtime/lane_c_range_model.zip)
    --trend-model  Path to trend-weighted model (default: runtime/lane_c_trend_model.zip)
    --dry-run      Log decisions but don't place orders
    --interval     Check interval in seconds (default: 60 = each M5 bar)
    --max-dd       Max drawdown % before shutdown (default: 5.0)
    --sl-atr       Stop-loss in ATR multiples (default: 2.0)

Environment variables:
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER — optional MT5 account credentials
"""
import sys, os, time, argparse, traceback, logging
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

# --- MT5 connection ---
try:
    import MetaTrader5 as mt5

    HAS_MT5 = True
except ImportError:
    HAS_MT5 = False
    mt5 = None

# --- Model infrastructure ---
from training.run_lane_c_mtf_regime import (
    LSTMFeatureExtractor,
    N_FEATURES,
    WINDOW_SIZE,
    HIDDEN_SIZE,
    N_LSTM_LAYERS,
    FEATURES_DIM,
    ADX_PERIOD,
    RANGE_THRESH,
    TREND_THRESH,
)

from gymnasium import spaces
from stable_baselines3 import PPO
import torch

# ---------------------------------------------------------------------------
# ADX regime detection (live-optimized, single-pass, standard Wilder's EMA)
# ---------------------------------------------------------------------------


def compute_adx_regime(highs, lows, closes):
    """Compute ADX(14) regime score using standard Wilder's smoothing (single pass).

    Returns (regime_score, adx_raw) both float32.
    """
    n = len(closes)
    if n < ADX_PERIOD * 2:
        return np.float32(0.0), np.float32(0.0)

    # True Range
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr[i] = max(hl, hc, lc)
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        if up > dn and up > 0:
            plus_dm[i] = up
        if dn > up and dn > 0:
            minus_dm[i] = dn

    # Standard Wilder's smoothing: separate EMA for TR, +DM, -DM
    alpha = 1.0 / ADX_PERIOD
    eps = 1e-10

    smooth_tr = np.mean(tr[1 : ADX_PERIOD + 1])
    smooth_pdm = np.mean(plus_dm[1 : ADX_PERIOD + 1])
    smooth_mdm = np.mean(minus_dm[1 : ADX_PERIOD + 1])

    # Collect DX values for ADX smoothing
    dx_history = []  # last ADX_PERIOD DX values for initial ADX calc

    for i in range(ADX_PERIOD + 1, n):
        smooth_tr = alpha * tr[i] + (1.0 - alpha) * smooth_tr
        smooth_pdm = alpha * plus_dm[i] + (1.0 - alpha) * smooth_pdm
        smooth_mdm = alpha * minus_dm[i] + (1.0 - alpha) * smooth_mdm

        pdi = 100.0 * smooth_pdm / max(smooth_tr, eps)
        mdi = 100.0 * smooth_mdm / max(smooth_tr, eps)
        ds = pdi + mdi
        dx = 100.0 * abs(pdi - mdi) / max(ds, eps)
        dx_history.append(dx)

    # ADX = EMA of DX (standard Wilder's)
    adx_raw = float(np.mean(dx_history[:ADX_PERIOD]))
    for j in range(ADX_PERIOD, len(dx_history)):
        adx_raw = alpha * dx_history[j] + (1.0 - alpha) * adx_raw

    # Regime score: ADX < 20 -> 0 (range), ADX > 40 -> 1 (trend)
    score = np.clip((adx_raw - 20.0) / 20.0, 0.0, 1.0)
    return np.float32(score), np.float32(adx_raw)


# ---------------------------------------------------------------------------
# Feature pipeline (matches training _build_features)
# ---------------------------------------------------------------------------

# Normalization stats frozen after warmup
_feature_means = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.35], dtype=np.float32
)
_feature_stds = np.array(
    [0.001, 0.001, 0.001, 0.001, 0.5, 28.0, 0.0005, 0.3], dtype=np.float32
)
_norm_frozen = False


def build_features(m5_df, regime_score):
    """Build 8 features for the latest bar from full-MA indicators.

    Uses full m5_df history for RSI and MACD (converges to training values).
    Returns (8,) float32 array.
    """
    o = m5_df["open"].values.astype(np.float64)
    h = m5_df["high"].values.astype(np.float64)
    l = m5_df["low"].values.astype(np.float64)
    c = m5_df["close"].values.astype(np.float64)
    v = m5_df["volume"].values.astype(np.float64)
    n = len(o)

    # OHLCV log returns for the LATEST bar
    ohlcv = np.zeros(5, dtype=np.float32)
    if n >= 2 and o[-2] > 0:
        ohlcv[0] = np.log(o[-1] / o[-2])
        ohlcv[1] = np.log(h[-1] / h[-2])
        ohlcv[2] = np.log(l[-1] / l[-2])
        ohlcv[3] = np.log(c[-1] / c[-2])
        ohlcv[4] = np.log(v[-1] / v[-2]) if v[-2] > 0 else 0.0

    # RSI(14) from full history
    delta = np.diff(c, prepend=c[0])
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    rsi_period = 14
    if n >= rsi_period:
        avg_gain = float(np.mean(gains[-rsi_period:]))
        avg_loss = float(np.mean(losses[-rsi_period:]))
        if avg_loss < 1e-10:
            rsi = 100.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    else:
        rsi = 50.0

    # MACD from full history (matches training: iterate ALL bars)
    alpha12 = 2.0 / 13.0
    alpha26 = 2.0 / 27.0
    alpha_sig = 2.0 / 10.0
    ema12 = c[0]
    ema26 = c[0]
    signal_line = 0.0
    for i in range(1, n):
        ema12 = alpha12 * c[i] + (1.0 - alpha12) * ema12
        ema26 = alpha26 * c[i] + (1.0 - alpha26) * ema26
        macd_line_i = ema12 - ema26
        if i == 1:
            signal_line = macd_line_i
        else:
            signal_line = alpha_sig * macd_line_i + (1.0 - alpha_sig) * signal_line
    macd_line = ema12 - ema26
    macd_hist_norm = float((macd_line - signal_line) / max(c[-1], 1.0))

    raw = np.array(
        [ohlcv[0], ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4], rsi, macd_hist_norm, float(regime_score)],
        dtype=np.float32,
    )
    return raw


def build_observation(feature_buffer):
    """Build (window_size * n_features,) flat observation from buffer.

    feature_buffer: (window_size, n_features) np.ndarray
    """
    global _feature_means, _feature_stds, _norm_frozen

    buf = feature_buffer.copy()

    if not _norm_frozen:
        # Compute stats from warmup data and freeze
        for col in range(N_FEATURES):
            col_data = buf[:, col]
            _feature_means[col] = float(np.mean(col_data))
            _feature_stds[col] = max(float(np.std(col_data)), 1e-10)
        _norm_frozen = True

    # Z-score normalize
    for col in range(N_FEATURES):
        buf[:, col] = (buf[:, col] - _feature_means[col]) / max(_feature_stds[col], 1e-10)

    return buf.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Meta-controller (mirrors training exactly)
# ---------------------------------------------------------------------------


def meta_controller_action(range_model, trend_model, obs, regime_score):
    """Get trading action (0=Long, 1=Flat, 2=Short) from meta-controller.

    Returns (action, diagnostics_dict).
    """
    range_action, _ = range_model.predict(obs, deterministic=True)
    trend_action, _ = trend_model.predict(obs, deterministic=True)

    ra = int(range_action.item()) if hasattr(range_action, "item") else int(range_action)
    ta = int(trend_action.item()) if hasattr(trend_action, "item") else int(trend_action)

    diag = {
        "range_raw": ra,
        "trend_raw": ta,
        "regime_score": float(regime_score),
        "blend_t": 0.0,
        "blended": 0.0,
        "decision": "range",
        "confidence": "high",
    }

    if regime_score < RANGE_THRESH:
        diag["decision"] = "range"
        diag["confidence"] = "high" if regime_score < (RANGE_THRESH - 0.1) else "medium"
        return ra, diag
    elif regime_score > TREND_THRESH:
        diag["decision"] = "trend"
        diag["confidence"] = "high" if regime_score > (TREND_THRESH + 0.1) else "medium"
        return ta, diag
    else:
        t = (regime_score - RANGE_THRESH) / (TREND_THRESH - RANGE_THRESH)
        diag["blend_t"] = round(t, 3)
        diag["decision"] = "blend"
        diag["confidence"] = "low"
        action_map = {0: 1.0, 1: 0.0, 2: -1.0}
        blended = (1.0 - t) * action_map.get(ra, 0.0) + t * action_map.get(ta, 0.0)
        diag["blended"] = round(blended, 3)
        if blended > 0.3:
            return 0, diag
        elif blended < -0.3:
            return 2, diag
        else:
            return 1, diag


# ---------------------------------------------------------------------------
# MT5 helpers
# ---------------------------------------------------------------------------


def mt5_connect():
    """Connect to MT5 using env vars or terminal defaults."""
    if not HAS_MT5 or mt5 is None:
        return False

    login = int(os.environ.get("MT5_LOGIN", 0))
    password = os.environ.get("MT5_PASSWORD", "")
    server = os.environ.get("MT5_SERVER", "")

    if login and password:
        ok = mt5.initialize(login=login, password=password, server=server)
    else:
        ok = mt5.initialize()

    if not ok:
        print(f"  MT5 init failed: {mt5.last_error()}")
        return False

    print(f"  MT5 connected: login={mt5.account_info().login}")
    return True


def fetch_bars(symbol, timeframe, count):
    """Fetch OHLCV bars from MT5."""
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "H1": mt5.TIMEFRAME_H1,
    }
    tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"Failed to fetch {timeframe} for {symbol}: {mt5.last_error()}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    if "real_volume" in df.columns:
        df["volume"] = df["real_volume"]
    elif "tick_volume" in df.columns and "volume" not in df.columns:
        df["volume"] = df["tick_volume"]
    return df


def get_atr(df, period=14):
    """Compute ATR from DataFrame."""
    h = df["high"].values.astype(np.float64)
    l = df["low"].values.astype(np.float64)
    c = df["close"].values.astype(np.float64)
    n = len(c)
    tr = np.zeros(n)
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = float(np.mean(tr[-period:])) if n >= period else float(np.mean(tr[1:]))
    return atr


def get_position(symbol):
    """Get current position. Returns (type, volume, open_price, ticket) or (None, 0, 0, 0)."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return None, 0.0, 0.0, 0
    pos = positions[0]
    return pos.type, pos.volume, pos.price_open, pos.ticket


def close_position(symbol):
    """Close all positions for symbol."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None or len(positions) == 0:
        return True
    for pos in positions:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            continue
        price = tick.bid if pos.type == 0 else tick.ask
        opposite = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": opposite,
            "position": pos.ticket,
            "price": price,
            "deviation": 20,
            "magic": 52042,
            "comment": "LaneC close",
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"  Close failed: retcode={result.retcode} {result.comment}")
            return False
    return True


def place_order(symbol, direction, volume, sl_price=0.0, comment="LaneC"):
    """Place a market order with optional stop-loss."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False
    if direction == "buy":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": sl_price if sl_price > 0 else 0.0,
        "deviation": 20,
        "magic": 52042,
        "comment": comment,
    }
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE, result


def calculate_position_size(symbol, risk_fraction):
    """Calculate position size based on risk fraction of equity."""
    account = mt5.account_info()
    if account is None:
        return 0.01
    equity = account.equity
    risk_amount = equity * risk_fraction

    info = mt5.symbol_info(symbol)
    if info is None:
        return 0.01

    # For XAUUSD: ~$2000 price, 100 oz per lot, 1 pip = $1 per 0.01 lot
    # Conservative: 0.01 lots per $1000 equity at 2% risk
    lots = max(0.01, round(risk_amount / 1000 * 0.01, 2))
    # Cap at reasonable maximum
    return min(lots, 1.0)


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Lane C Live Trading Bot")
    parser.add_argument("--symbol", default="XAUUSDm", help="Trading symbol")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk fraction of equity")
    parser.add_argument("--range-model", default="runtime/lane_c_range_model.zip")
    parser.add_argument("--trend-model", default="runtime/lane_c_trend_model.zip")
    parser.add_argument("--dry-run", action="store_true", help="Log only, no real orders")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--max-dd", type=float, default=5.0, help="Max drawdown %% before shutdown")
    parser.add_argument("--sl-atr", type=float, default=2.0, help="Stop-loss in ATR multiples")
    parser.add_argument("--h1-bars", type=int, default=500, help="H1 bars for regime detection")
    parser.add_argument("--max-bars", type=int, default=0, help="Exit after N bars (0=run forever, for dry-run testing)")
    args = parser.parse_args()

    print("=" * 64)
    print("LANE C LIVE TRADING BOT")
    print(f"  Symbol:   {args.symbol}")
    print(f"  Risk:     {args.risk * 100:.1f}%")
    print(f"  Max DD:   {args.max_dd:.1f}%")
    print(f"  SL ATR:   {args.sl_atr}x")
    print(f"  Dry run:  {args.dry_run}")
    print("=" * 64)
    print()

    # --- Connect to MT5 ---
    if not HAS_MT5:
        print("ERROR: MetaTrader5 not installed.")
        return
    # Always connect to MT5 for data — dry-run only skips order placement
    if not mt5_connect():
        print("ERROR: Could not connect to MT5. Is the terminal running?")
        return

    # --- Load models ---
    for model_path in [args.range_model, args.trend_model]:
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            print("Run: python training/run_lane_c_mtf_regime.py --steps 50000")
            return

    print("  Loading models...")
    model_range = PPO.load(args.range_model)
    model_trend = PPO.load(args.trend_model)
    print(f"  Range model  loaded: {args.range_model}")
    print(f"  Trend model  loaded: {args.trend_model}")
    print()

    # --- Warmup ---
    print("  Warming up (fetching initial data)...")
    try:
        m5_df = fetch_bars(args.symbol, "M5", WINDOW_SIZE + 200)
        h1_df = fetch_bars(args.symbol, "H1", args.h1_bars)
    except Exception as e:
        print(f"  ERROR: {e}")
        if not args.dry_run:
            mt5.shutdown()
        return

    # Build feature buffer and fill from history
    feature_buffer = np.zeros((WINDOW_SIZE, N_FEATURES), dtype=np.float32)

    # Compute initial H1 regime
    h1_high = h1_df["high"].values.astype(np.float64)
    h1_low = h1_df["low"].values.astype(np.float64)
    h1_close = h1_df["close"].values.astype(np.float64)
    regime_score, adx_value = compute_adx_regime(h1_high, h1_low, h1_close)

    # Fill feature buffer from recent M5 bars
    for i in range(WINDOW_SIZE):
        idx = len(m5_df) - WINDOW_SIZE + i
        feat = build_features(m5_df.iloc[max(0, idx - WINDOW_SIZE) : idx + 1], regime_score)
        feature_buffer[i] = feat

    # Freeze normalization
    obs = build_observation(feature_buffer)

    zone = "RANGE" if regime_score < RANGE_THRESH else ("TREND" if regime_score > TREND_THRESH else "TRANSITION")
    print(f"  H1 regime: score={regime_score:.3f}, ADX={adx_value:.1f}, zone={zone}")
    print(f"  Norm stats frozen: means={_feature_means}, stds={_feature_stds}")
    print()

    # --- Start trading ---
    print("  Trading loop started...")
    print(f"  {'Time':<10} {'Zone':<12} {'Action':<8} {'Pos':<8} {'Score':>6} {'ADX':>6} {'Equity':>10} {'DD%':>6}")
    print(f"  {'-'*10} {'-'*12} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*10} {'-'*6}")

    last_bar_time = None
    current_position = "FLAT"
    peak_equity = 0.0
    start_equity = 0.0
    consecutive_errors = 0
    bar_count = 0
    MAX_CONSECUTIVE_ERRORS = 5

    if not args.dry_run:
        account = mt5.account_info()
        peak_equity = account.equity if account else 0.0
        start_equity = peak_equity

    try:
        while True:
            bar_count += 1
            if args.max_bars > 0 and bar_count > args.max_bars:
                print(f"\n  Reached {args.max_bars} decision cycles, exiting cleanly.")
                break

            now = datetime.now(timezone.utc)

            # Fetch latest data
            try:
                m5_df = fetch_bars(args.symbol, "M5", WINDOW_SIZE + 50)
                h1_df = fetch_bars(args.symbol, "H1", args.h1_bars)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                print(f"  [{now:%H:%M:%S}] Fetch error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    print("  Too many errors, shutting down.")
                    break
                time.sleep(10)
                continue

            # Check for new bar
            latest_bar = m5_df["time"].iloc[-1]
            if last_bar_time is not None and latest_bar <= last_bar_time:
                time.sleep(5)
                continue
            last_bar_time = latest_bar

            # --- Update account metrics ---
            if not args.dry_run:
                account = mt5.account_info()
                equity = account.equity if account else 0.0
                if equity > peak_equity:
                    peak_equity = equity
                dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
                if dd_pct >= args.max_dd:
                    print(f"  MAX DRAWDOWN ({dd_pct:.1f}% >= {args.max_dd}%), shutting down.")
                    break
            else:
                equity = 0.0
                dd_pct = 0.0

            # --- Compute H1 regime ---
            h1_high2 = h1_df["high"].values.astype(np.float64)
            h1_low2 = h1_df["low"].values.astype(np.float64)
            h1_close2 = h1_df["close"].values.astype(np.float64)
            regime_score, adx_value = compute_adx_regime(h1_high2, h1_low2, h1_close2)
            zone = "RANGE" if regime_score < RANGE_THRESH else ("TREND" if regime_score > TREND_THRESH else "TRANSITION")

            # --- Build features ---
            feature_buffer[:-1] = feature_buffer[1:]
            feat = build_features(m5_df, regime_score)
            feature_buffer[-1] = feat
            obs = build_observation(feature_buffer)

            # --- Get action ---
            action, diag = meta_controller_action(model_range, model_trend, obs, regime_score)
            desired = {0: "LONG", 1: "FLAT", 2: "SHORT"}.get(action, "FLAT")

            # Get spread (with fallback for MT5 hiccups)
            try:
                tick = mt5.symbol_info_tick(args.symbol)
                spread = (tick.ask - tick.bid) if tick else 0.0
            except Exception:
                tick = None
                spread = 0.0
            spread_pts = int(spread * 10) if spread > 0 else 0  # XAUUSD points

            # --- Execute trade ---
            if desired != current_position:
                resolution = ""
                if args.dry_run:
                    if current_position != "FLAT":
                        print(f"  [{now:%H:%M:%S}] [DRY] CLOSE {current_position} -> {desired}")
                    else:
                        print(f"  [{now:%H:%M:%S}] [DRY] ENTER {desired}")
                    resolution = f"  (DRY) {current_position}->{desired}"
                else:
                    # Close existing
                    if current_position != "FLAT":
                        if not close_position(args.symbol):
                            print(f"  [{now:%H:%M:%S}] CLOSE {current_position} FAILED")
                        else:
                            print(f"  [{now:%H:%M:%S}] CLOSED {current_position}")

                    # Open new
                    if desired != "FLAT":
                        vol = calculate_position_size(args.symbol, args.risk)
                        tick = mt5.symbol_info_tick(args.symbol)
                        atr = get_atr(m5_df)
                        sl_dist = atr * args.sl_atr  # price units
                        dir_str = "buy" if desired == "LONG" else "sell"
                        # Stop-loss = entry price +/- ATR * multiplier
                        sl_price = (tick.ask - sl_dist) if dir_str == "buy" else (tick.bid + sl_dist)
                        comment = f"LaneC_{zone}"
                        ok, result = place_order(args.symbol, dir_str, vol, sl_price=sl_price, comment=comment)
                        if ok:
                            print(
                                f"  [{now:%H:%M:%S}] ENTER {desired} vol={vol} zone={zone} "
                                f"SL={sl_price:.2f}"
                            )
                            resolution = f"  ENTER {desired} @ {vol}"
                        else:
                            print(f"  [{now:%H:%M:%S}] ENTER {desired} FAILED: {result.comment}")
                            resolution = f"  ORDER FAILED: {result.comment}"

                current_position = desired

            # Log: standard line + detailed diagnostics
            print(
                f"  [{now:%H:%M:%S}] {zone:<12} {desired:<8} {current_position:<8} "
                f"{regime_score:.3f} {adx_value:6.1f} {equity:>10.1f} {dd_pct:>5.1f}"
            )
            # Detailed decision log
            raw_label = {0: "LONG", 1: "FLAT", 2: "SHORT"}
            print(
                f"         [DIAG] decision={diag['decision']} conf={diag['confidence']} "
                f"range_raw={raw_label.get(diag['range_raw'], '?')} "
                f"trend_raw={raw_label.get(diag['trend_raw'], '?')} "
                f"blend_t={diag.get('blend_t', 0):.3f} "
                f"blended={diag.get('blended', 0):.3f} "
                f"spread={spread_pts}pts"
                + (f" dry-run" if args.dry_run else "")
            )

            # Wait for next bar (or shorter interval in dry-run)
            time.sleep(max(1, args.interval - 5) if args.dry_run else max(5, args.interval - 5))

    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        traceback.print_exc()
    finally:
        if not args.dry_run and HAS_MT5:
            print("  Closing position...")
            close_position(args.symbol)
            mt5.shutdown()
            print("  MT5 shut down.")
        print("  Done.")


if __name__ == "__main__":
    main()
