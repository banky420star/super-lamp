"""
Lane B Live Trading Bot — Single LSTM PPO Model (Discrete Actions).

Connects to MT5, loads a trained Lane B model (seed 456 champion),
fetches live M5 bars, builds 7 OHLCV+RSI+MACD features, and executes
trades via direct PPO discrete action prediction.

Usage
-----
    # First, train and save a model:
    python training/run_lane_b_raw_lstm.py --steps 50000 --seed 456

    # Then run live trading:
    python training/live_trade_lane_b.py --symbol XAUUSDm --risk 0.02

    # Dry-run (no real orders, just log decisions):
    python training/live_trade_lane_b.py --dry-run

CLI args:
    --symbol       Trading symbol (default: XAUUSDm)
    --risk         Risk per trade as fraction of equity (default: 0.02 = 2%)
    --model        Path to Lane B model (default: runtime/lane_b_seed_456_model.zip)
    --dry-run      Log decisions but don't place orders
    --interval     Check interval in seconds (default: 60 = each M5 bar)
    --max-dd       Max drawdown % before shutdown (default: 5.0)
    --sl-atr       Stop-loss in ATR multiples (default: 2.0)
    --max-bars     Exit after N decision cycles (0=run forever, for testing)
    --h1-bars      H1 bars for warmup display only (default: 300)

Environment variables:
    MT5_LOGIN, MT5_PASSWORD, MT5_SERVER — optional MT5 account credentials
    TG_BOT_TOKEN   — Telegram bot token from @BotFather (optional)
    TG_CHAT_ID     — Telegram chat ID (optional)
"""
import sys, os, time, argparse, traceback, csv, json
from datetime import datetime, timezone

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
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn

# --- Telegram alerts ---
try:
    from training.telegram_alerts import (
        telegram_available,
        send_alert,
        send_trade_alert,
        send_close_alert,
        send_dd_alert,
        send_startup_alert,
        send_shutdown_alert,
    )

    HAS_TELEGRAM = True
except ImportError:
    HAS_TELEGRAM = False

# ── Lane B constants (must match run_lane_b_raw_lstm.py exactly) ──
N_FEATURES = 7          # 5 OHLCV log-returns + 1 RSI + 1 MACD histogram
WINDOW_SIZE = 64        # 64 x 5min = 320min ~ 5.3 hours of context
HIDDEN_SIZE = 128
N_LSTM_LAYERS = 2
FEATURES_DIM = 64


# ---------------------------------------------------------------------------
# LSTM Feature Extractor (must match run_lane_b_raw_lstm.py exactly)
# ---------------------------------------------------------------------------

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Two-layer LSTM that matches Lane B training architecture."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
        self.window_size = WINDOW_SIZE
        self.n_features = N_FEATURES

        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=N_LSTM_LAYERS,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if N_LSTM_LAYERS > 1 else 0,
        )
        self.projection = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, features_dim),
            nn.LayerNorm(features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.window_size, self.n_features)
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        return self.projection(last)


# ---------------------------------------------------------------------------
# Feature pipeline (matches TamedOHLCVEnv._build_features exactly)
# ---------------------------------------------------------------------------

# Normalization stats frozen after warmup
_feature_means = np.array([0.0] * N_FEATURES, dtype=np.float32)
_feature_stds = np.array([1.0] * N_FEATURES, dtype=np.float32)
_norm_frozen = False


def build_features(m5_df):
    """Build 7 features for the latest bar (matches _build_features from training).

    Features (column order):
        0-4: OHLCV log-returns (latest bar vs previous)
        5:   RSI(14) from full history
        6:   MACD histogram / price from full history

    Returns (7,) float32 array.
    """
    o = m5_df["open"].values.astype(np.float64)
    h = m5_df["high"].values.astype(np.float64)
    l = m5_df["low"].values.astype(np.float64)
    c = m5_df["close"].values.astype(np.float64)
    v = m5_df["volume"].values.astype(np.float64)
    n = len(o)

    # ── OHLCV log-returns for the LATEST bar (columns 0-4) ──
    ohlcv = np.zeros(5, dtype=np.float32)
    if n >= 2 and o[-2] > 0:
        ohlcv[0] = np.log(o[-1] / o[-2])
        ohlcv[1] = np.log(h[-1] / h[-2])
        ohlcv[2] = np.log(l[-1] / l[-2])
        ohlcv[3] = np.log(c[-1] / c[-2])
        ohlcv[4] = np.log(v[-1] / v[-2]) if v[-2] > 0 else 0.0

    # ── RSI(14) from full history (column 5) ──
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

    # ── MACD histogram from full history (column 6, matches training: iterate ALL bars) ──
    alpha12 = 2.0 / 13.0
    alpha26 = 2.0 / 27.0
    alpha_sig = 2.0 / 10.0
    ema12 = c[0]
    ema26 = c[0]
    # Training: signal_line[0] = macd_line[0] = ema12[0] - ema26[0] = c[0] - c[0] = 0
    signal_line = 0.0
    for i in range(1, n):
        ema12 = alpha12 * c[i] + (1.0 - alpha12) * ema12
        ema26 = alpha26 * c[i] + (1.0 - alpha26) * ema26
        macd_line_i = ema12 - ema26
        signal_line = alpha_sig * macd_line_i + (1.0 - alpha_sig) * signal_line
    macd_line = ema12 - ema26
    macd_hist_norm = float((macd_line - signal_line) / max(c[-1], 1.0))

    raw = np.array(
        [ohlcv[0], ohlcv[1], ohlcv[2], ohlcv[3], ohlcv[4],
         rsi, macd_hist_norm],
        dtype=np.float32,
    )
    return raw


def build_observation(feature_buffer):
    """Build (window_size * n_features,) flat observation from buffer.

    Z-score normalizes using stats frozen during warmup.
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

    account = mt5.account_info()
    if account is None:
        print("  MT5 account_info() returned None")
        return False

    print(f"  MT5 connected: login={account.login}, server={account.server}")
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
    if n < 2:
        return 0.0
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
            "magic": 52043,
            "comment": "LaneB close",
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"  Close failed: retcode={result.retcode} {result.comment}")
            return False
    return True


def place_order(symbol, direction, volume, sl_price=0.0, comment="LaneB"):
    """Place a market order with optional stop-loss."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False, None
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
        "magic": 52043,
        "comment": comment,
    }
    result = mt5.order_send(request)
    return result.retcode == mt5.TRADE_RETCODE_DONE, result


def is_market_open(symbol):
    """Check if the symbol's market is currently open for trading.

    Uses mt5.symbol_info().time (last quote timestamp) to detect weekends
    and closed sessions. A quote older than 4 hours means the market is
    closed. Falls back to tick bid/ask check if session info is unavailable.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        tick = mt5.symbol_info_tick(symbol)
        return tick is not None and tick.bid > 0 and tick.ask > 0

    # trade_mode == 0 means SYMBOL_TRADE_MODE_DISABLED
    if info.trade_mode == 0:
        return False

    # Primary check: last quote timestamp from symbol_info()
    # During weekends / closed sessions, info.time is hours or days old
    if info.time and info.time > 0:
        last_quote = datetime.fromtimestamp(info.time, tz=timezone.utc)
        age_seconds = (datetime.now(timezone.utc) - last_quote).total_seconds()
        STALE_THRESHOLD = 4 * 3600  # 4 hours
        if age_seconds > STALE_THRESHOLD:
            return False
        return True

    # Fallback: tick check
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return False
    return tick.bid > 0 and tick.ask > 0 and tick.bid != tick.ask


def wait_for_market(symbol, timeout_minutes=720):
    """Block until the market opens. Prints status every 60s. Exits after timeout."""
    print(f"  Market appears closed for {symbol}. Waiting for open (timeout={timeout_minutes}min)...")
    check_interval = 60
    waited = 0
    timeout_sec = timeout_minutes * 60
    while True:
        if is_market_open(symbol):
            print(f"  Market opened after {waited // 60}min.")
            return True
        if waited >= timeout_sec:
            now = datetime.now(timezone.utc)
            print(f"  [{now:%H:%M:%S}] Market still closed after {waited // 60}min. Timeout reached.")
            return False
        if waited == 0 or waited % 300 == 0:
            now = datetime.now(timezone.utc)
            remaining = (timeout_sec - waited) // 60
            print(f"  [{now:%H:%M:%S}] Market closed ({waited // 60}min waited, {remaining}min until timeout)...")
        time.sleep(check_interval)
        waited += check_interval


def log_closed_trade(symbol, direction, open_time, close_time, open_price, close_price,
                     volume, profit, ticket, model_name, exit_reason):
    """Append a closed trade to the per-symbol PnL CSV journal.

    The CSV is stored at runtime/lane_b_{symbol}_trades.csv and is independent
    of MT5 history — it survives terminal restarts and account changes.
    """
    os.makedirs("runtime", exist_ok=True)
    csv_path = f"runtime/lane_b_{symbol}_trades.csv"
    file_exists = os.path.exists(csv_path)
    now = datetime.now(timezone.utc).isoformat()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "logged_at", "symbol", "direction", "open_time", "close_time",
                "open_price", "close_price", "volume", "profit", "ticket",
                "model", "exit_reason"
            ])
        writer.writerow([
            now, symbol, direction,
            open_time.isoformat() if open_time else "",
            close_time.isoformat() if close_time else "",
            round(open_price, 5) if open_price else 0,
            round(close_price, 5) if close_price else 0,
            volume, round(profit, 2), ticket,
            model_name, exit_reason,
        ])


def calculate_position_size(symbol, risk_fraction):
    """Calculate position size based on risk fraction of equity."""
    account = mt5.account_info()
    if account is None:
        return 0.01
    equity = account.equity
    risk_amount = equity * risk_fraction

    # For XAUUSD: ~$2000 price, 100 oz per lot, 1 pip = $1 per 0.01 lot
    # Conservative: 0.01 lots per $1000 equity at 2% risk
    lots = max(0.01, round(risk_amount / 1000 * 0.01, 2))
    return min(lots, 1.0)


# ---------------------------------------------------------------------------
# Main trading loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lane B Live Trading Bot")
    parser.add_argument("--symbol", default="XAUUSDm", help="Trading symbol")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk fraction of equity")
    parser.add_argument("--model", default="runtime/lane_b_seed_456_model.zip",
                        help="Path to Lane B model")
    parser.add_argument("--dry-run", action="store_true", help="Log only, no real orders")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--max-dd", type=float, default=5.0, help="Max drawdown %% before shutdown")
    parser.add_argument("--sl-atr", type=float, default=2.0, help="Stop-loss in ATR multiples")
    parser.add_argument("--h1-bars", type=int, default=300, help="H1 bars for warmup display")
    parser.add_argument("--max-bars", type=int, default=0,
                        help="Exit after N decision cycles (0=run forever)")
    parser.add_argument("--telegram", action="store_true",
                        help="Enable Telegram alerts (requires TG_BOT_TOKEN and TG_CHAT_ID env vars)")
    args = parser.parse_args()

    print("=" * 64)
    print("LANE B LIVE TRADING BOT")
    print(f"  Symbol:   {args.symbol}")
    print(f"  Risk:     {args.risk * 100:.1f}%")
    print(f"  Max DD:   {args.max_dd:.1f}%")
    print(f"  SL ATR:   {args.sl_atr}x")
    print(f"  Model:    {args.model}")
    print(f"  Dry run:  {args.dry_run}")
    print(f"  Features: {N_FEATURES} (OHLCV + RSI + MACD)")
    print(f"  Window:   {WINDOW_SIZE} bars")
    print(f"  Action:   Discrete(3) [Long/Flat/Short]")
    print("=" * 64)
    print()

    # --- Connect to MT5 ---
    if not HAS_MT5:
        print("ERROR: MetaTrader5 not installed.")
        return
    if not mt5_connect():
        print("ERROR: Could not connect to MT5. Is the terminal running?")
        return

    # --- Check market open (skip in dry-run) ---
    if not args.dry_run:
        if not is_market_open(args.symbol):
            if not wait_for_market(args.symbol):
                print(f"  Market timeout for {args.symbol}. Exiting.")
                mt5.shutdown()
                return
        else:
            print(f"  Market open for {args.symbol}")

    # --- Load model ---
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        print("Run: python training/run_lane_b_raw_lstm.py --steps 50000 --seed 456")
        return

    print("  Loading model...")
    model = PPO.load(args.model)
    print(f"  Model loaded: {args.model}")
    print()

    # --- Check Telegram ---
    tg_ok = False
    if args.telegram:
        if HAS_TELEGRAM and telegram_available():
            tg_ok = True
            print("  Telegram: enabled")
        else:
            print("  Telegram: unavailable (set TG_BOT_TOKEN and TG_CHAT_ID env vars)")

    # --- Warmup ---
    print("  Warming up (fetching initial data)...")
    try:
        m5_df = fetch_bars(args.symbol, "M5", WINDOW_SIZE + 200)
        # Also fetch H1 bars for display context (not used by Lane B)
        h1_df = fetch_bars(args.symbol, "H1", args.h1_bars)
    except Exception as e:
        print(f"  ERROR: {e}")
        if not args.dry_run:
            mt5.shutdown()
        return

    print(f"  M5 bars loaded: {len(m5_df)}")
    print(f"  H1 bars loaded: {len(h1_df)}")

    # Build feature buffer and fill from history
    feature_buffer = np.zeros((WINDOW_SIZE, N_FEATURES), dtype=np.float32)

    # Fill feature buffer from recent M5 bars
    # Use full history up to each bar so MACD/RSI have consistent context
    for i in range(WINDOW_SIZE):
        idx = len(m5_df) - WINDOW_SIZE + i
        feat = build_features(m5_df.iloc[: idx + 1])
        feature_buffer[i] = feat

    # Freeze normalization
    obs = build_observation(feature_buffer)

    print(f"  Norm stats frozen: means={np.round(_feature_means, 6)}")
    print(f"                     stds={np.round(_feature_stds, 6)}")
    print()

    # Display H1 context (for informational purposes only — Lane B doesn't use regimes)
    try:
        h1_high = h1_df["high"].values.astype(np.float64)
        h1_low = h1_df["low"].values.astype(np.float64)
        h1_close = h1_df["close"].values.astype(np.float64)
        h1_range = float(h1_high.max() - h1_low.min())
        h1_trend = "UP" if h1_close[-1] > h1_close[-20] else "DOWN"
        print(f"  H1 context: range={h1_range:.2f}, last={h1_close[-1]:.2f}, "
              f"trend(20)={h1_trend}")
    except Exception:
        print("  H1 context: unavailable")
    print()

    # --- Start trading ---
    print("  Trading loop started...")
    print(f"  {'Time':<10} {'Action':<8} {'Pos':<8} {'Conf':>6} {'Equity':>10} {'DD%':>6}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*6} {'-'*10} {'-'*6}")

    last_bar_time = None
    current_position = "FLAT"
    peak_equity = 0.0
    start_equity = 0.0
    consecutive_errors = 0
    bar_count = 0
    MAX_CONSECUTIVE_ERRORS = 5

    trades_taken = 0
    last_ticket = 0  # track last position ticket for PnL lookup on close
    last_entry_open_time = None  # position open time for journal
    last_entry_open_price = 0.0  # position open price for journal
    last_entry_volume = 0.0       # position volume for journal
    equity_last_report = 0.0
    equity_report_interval = 60  # report equity every N bars

    if not args.dry_run:
        account = mt5.account_info()
        peak_equity = account.equity if account else 0.0
        start_equity = peak_equity
        if tg_ok:
            send_startup_alert(args.symbol, os.path.basename(args.model),
                               args.risk * 100, start_equity)

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
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                print(f"  [{now:%H:%M:%S}] Fetch error ({consecutive_errors}/"
                      f"{MAX_CONSECUTIVE_ERRORS}): {e}")
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

            # --- Reconcile actual MT5 position with internal state ---
            # Fixes state drift when SL is hit externally or position closes
            # without bot noticing. Runs every cycle regardless of model decision.
            if not args.dry_run:
                try:
                    live_pos = mt5.positions_get(symbol=args.symbol)
                    has_position = live_pos is not None and len(live_pos) > 0
                    if current_position != "FLAT" and not has_position:
                        print(
                            f"  [{now:%H:%M:%S}] WARNING: Position lost externally"
                            f" (SL/TP hit?). Resetting {current_position} -> FLAT"
                        )
                        # Log the externally-closed trade to CSV journal
                        try:
                            deals = mt5.history_deals_get(position=last_ticket)
                            profit = sum(d.profit for d in deals) if deals else 0.0
                        except Exception:
                            profit = 0.0
                        log_closed_trade(
                            symbol=args.symbol,
                            direction=current_position,
                            open_time=last_entry_open_time,
                            close_time=now,
                            open_price=last_entry_open_price,
                            close_price=0,
                            volume=last_entry_volume,
                            profit=profit,
                            ticket=last_ticket,
                            model_name=os.path.basename(args.model),
                            exit_reason="external",
                        )
                        current_position = "FLAT"
                        last_ticket = 0
                        last_entry_open_time = None
                        last_entry_open_price = 0.0
                    elif current_position == "FLAT" and has_position:
                        p = live_pos[0]
                        actual_type = "LONG" if p.type == 0 else "SHORT"
                        print(
                            f"  [{now:%H:%M:%S}] WARNING: Found unexpected"
                            f" {actual_type} position (ticket={p.ticket})."
                            f" Adopting it."
                        )
                        current_position = actual_type
                        last_ticket = p.ticket
                        last_entry_open_price = p.price_open
                        last_entry_open_time = datetime.fromtimestamp(p.time, tz=timezone.utc) if p.time else None
                        last_entry_volume = p.volume
                except Exception as pos_recon_err:
                    print(f"  [{now:%H:%M:%S}] Position reconciliation error: {pos_recon_err}")

            # --- Update account metrics ---
            if not args.dry_run:
                account = mt5.account_info()
                equity = account.equity if account else 0.0
                if equity > peak_equity:
                    peak_equity = equity
                dd_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
                if dd_pct >= args.max_dd:
                    print(f"  MAX DRAWDOWN ({dd_pct:.1f}% >= {args.max_dd}%), shutting down.")
                    if tg_ok:
                        send_dd_alert(args.symbol, dd_pct, args.max_dd, equity)
                    break
            else:
                equity = 0.0
                dd_pct = 0.0

            # --- Build features ---
            feature_buffer[:-1] = feature_buffer[1:]
            feat = build_features(m5_df)
            feature_buffer[-1] = feat
            obs = build_observation(feature_buffer)

            # --- Get model action ---
            # Discrete(3): 0=Long, 1=Flat, 2=Short
            action_raw, _states = model.predict(obs, deterministic=True)
            action = int(action_raw.item()) if hasattr(action_raw, "item") else int(action_raw)
            desired = {0: "LONG", 1: "FLAT", 2: "SHORT"}.get(action, "FLAT")

            # Confidence proxy: std of last-layer logits (higher = more decisive)
            # Not directly available from PPO.predict, use action value as proxy
            conf_label = "OK"

            # Get spread
            try:
                tick = mt5.symbol_info_tick(args.symbol)
                spread = (tick.ask - tick.bid) if tick else 0.0
                spread_pts = int(spread * 10) if spread > 0 else 0
            except Exception:
                tick = None
                spread = 0.0
                spread_pts = 0

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
                            # Get PnL from last tracked ticket
                            profit = 0.0
                            if last_ticket > 0:
                                deals = mt5.history_deals_get(position=last_ticket)
                                profit = sum(d.profit for d in deals) if deals else 0.0
                            # Look up close price from deal history
                            close_price = 0.0
                            if last_ticket > 0:
                                try:
                                    deal_pos = mt5.history_deals_get(position=last_ticket)
                                    if deal_pos:
                                        close_price = deal_pos[-1].price
                                except Exception:
                                    pass
                            # Log to CSV journal
                            log_closed_trade(
                                symbol=args.symbol,
                                direction=current_position,
                                open_time=last_entry_open_time,
                                close_time=now,
                                open_price=last_entry_open_price,
                                close_price=close_price,
                                volume=last_entry_volume,
                                profit=profit,
                                ticket=last_ticket,
                                model_name=os.path.basename(args.model),
                                exit_reason="signal",
                            )
                            if tg_ok:
                                send_close_alert(args.symbol, current_position, profit)
                            last_ticket = 0
                            last_entry_open_time = None
                            last_entry_open_price = 0.0
                            last_entry_volume = 0.0

                    # Open new
                    if desired != "FLAT":
                        vol = calculate_position_size(args.symbol, args.risk)
                        tick2 = mt5.symbol_info_tick(args.symbol)
                        atr = get_atr(m5_df)
                        sl_dist = atr * args.sl_atr
                        dir_str = "buy" if desired == "LONG" else "sell"
                        sl_price = (tick2.ask - sl_dist) if dir_str == "buy" else (tick2.bid + sl_dist)
                        comment = f"LaneB_{os.path.basename(args.model)[:8]}"
                        ok, result = place_order(args.symbol, dir_str, vol,
                                                 sl_price=sl_price, comment=comment)
                        if ok:
                            print(
                                f"  [{now:%H:%M:%S}] ENTER {desired} vol={vol} "
                                f"SL={sl_price:.2f}"
                            )
                            if tg_ok:
                                send_trade_alert(args.symbol, desired, vol,
                                                 tick2.ask if dir_str == "buy" else tick2.bid,
                                                 sl_price, comment)
                            trades_taken += 1
                            # Record entry details for journal
                            live_pos = mt5.positions_get(symbol=args.symbol)
                            if live_pos:
                                last_ticket = live_pos[0].ticket
                                last_entry_open_time = now
                                last_entry_open_price = live_pos[0].price_open
                                last_entry_volume = vol
                            else:
                                last_ticket = 0
                                last_entry_open_time = None
                                last_entry_open_price = 0.0
                                last_entry_volume = 0.0
                            resolution = f"  ENTER {desired} @ {vol}"
                        else:
                            print(f"  [{now:%H:%M:%S}] ENTER {desired} FAILED: "
                                  f"{result.comment if result else 'unknown'}")
                            resolution = f"  ORDER FAILED"

                current_position = desired

            # Log: standard line + detailed diagnostics
            print(
                f"  [{now:%H:%M:%S}] {desired:<8} {current_position:<8} "
                f"{conf_label:>6} {equity:>10.1f} {dd_pct:>5.1f}"
            )
            # Detailed decision log
            action_label = {0: "LONG", 1: "FLAT", 2: "SHORT"}
            print(
                f"         [DIAG] action={action_label.get(action, '?')} "
                f"spread={spread_pts}pts"
                + (f" dry-run" if args.dry_run else "")
            )

            # ── Write status file for dashboard (per-symbol) ──
            os.makedirs("runtime", exist_ok=True)
            status = {
                "symbol": args.symbol,
                "dry_run": args.dry_run,
                "risk_pct": args.risk,
                "max_dd_pct": args.max_dd,
                "sl_atr": args.sl_atr,
                "interval_sec": args.interval,
                "current_position": current_position,
                "last_action": action,
                "bar_count": bar_count,
                "trades_taken": trades_taken,
                "peak_equity": peak_equity,
                "start_equity": start_equity,
                "drawdown_pct": dd_pct,
                "model_name": os.path.basename(args.model),
                "pid": os.getpid(),
            }
            status_path = f"runtime/lane_b_{args.symbol}_status.json"
            with open(status_path, "w") as sf:
                json.dump(status, sf)

            # Periodic equity report every N cycles
            if tg_ok and not args.dry_run and bar_count % equity_report_interval == 0:
                pnl = equity - start_equity
                sign = "+" if pnl >= 0 else ""
                send_alert(
                    f"📊 <b>Equity Update</b> (cycle {bar_count})\n"
                    f"Balance={equity:.2f}  PnL={sign}{pnl:.2f}\n"
                    f"DD={dd_pct:.1f}%  Trades={trades_taken}",
                    silent=True
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
            final_account = mt5.account_info()
            final_equity = final_account.equity if final_account else 0.0
            if tg_ok:
                send_shutdown_alert(args.symbol, start_equity, final_equity, trades_taken)
            mt5.shutdown()
            print("  MT5 shut down.")
        print("  Done.")


if __name__ == "__main__":
    main()
