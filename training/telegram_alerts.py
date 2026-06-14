"""
Telegram alerts helper for trading bots.

Sends messages via the Telegram Bot API using `requests`.
Reads TG_BOT_TOKEN and TG_CHAT_ID from environment variables.

Usage:
    from training.telegram_alerts import send_alert, telegram_available

    if telegram_available():
        send_alert("LANE B: ENTER LONG vol=0.05 @ 2350.50 SL=2345.00")

Environment variables:
    TG_BOT_TOKEN   — Telegram bot token from @BotFather
    TG_CHAT_ID     — Your chat ID (numeric, e.g. 123456789)

To set up:
    1. Message @BotFather on Telegram and create a new bot
    2. Copy the API token
    3. Message your bot once, then visit:
       https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
    4. Find your chat_id in the response
    5. Set env vars:
       set TG_BOT_TOKEN=your_token_here
       set TG_CHAT_ID=your_chat_id_here
"""
import os
import logging

logger = logging.getLogger(__name__)

# Lazy imports — don't require requests at module load
_requests = None


def _import_requests():
    """Import requests lazily."""
    global _requests
    if _requests is None:
        try:
            import requests as req

            _requests = req
        except ImportError:
            _requests = False
    return _requests if _requests is not False else None


def telegram_available():
    """Check if Telegram credentials and requests are available."""
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat_id = os.environ.get("TG_CHAT_ID", "")
    req = _import_requests()
    return bool(token and chat_id and req is not None)


def send_alert(message, silent=False):
    """Send a Telegram alert message.

    Args:
        message:  Text message to send (max 4096 chars)
        silent:   If True, sends with disable_notification (no sound)

    Returns:
        True if sent successfully, False otherwise.
    """
    token = os.environ.get("TG_BOT_TOKEN", "")
    chat_id = os.environ.get("TG_CHAT_ID", "")
    req = _import_requests()

    if not token or not chat_id or req is None:
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message[:4000],  # Telegram limit is 4096
        "parse_mode": "HTML",
        "disable_notification": silent,
    }

    try:
        resp = req.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            return True
        else:
            logger.warning(f"Telegram send failed: HTTP {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        logger.warning(f"Telegram send error: {e}")
        return False


def send_trade_alert(symbol, direction, volume, price, sl_price=None, comment=""):
    """Send a formatted trade entry alert."""
    emoji = "🟢" if direction == "LONG" else "🔴"
    sl = f" SL={sl_price:.2f}" if sl_price else ""
    msg = (
        f"{emoji} <b>{direction}</b> {symbol}\n"
        f"Vol={volume:.2f}  Price={price:.2f}{sl}\n"
        f"Bot={comment}"
    )
    return send_alert(msg)


def send_close_alert(symbol, direction, profit, comment=""):
    """Send a formatted trade close alert."""
    emoji = "✅" if profit >= 0 else "❌"
    sign = "+" if profit >= 0 else ""
    msg = (
        f"{emoji} <b>CLOSED {direction}</b> {symbol}\n"
        f"PnL={sign}{profit:.2f}  {comment}"
    )
    return send_alert(msg)


def send_dd_alert(symbol, dd_pct, max_dd, equity):
    """Send a max drawdown warning alert."""
    msg = (
        f"⚠️ <b>MAX DRAWDOWN</b>\n"
        f"{symbol}  DD={dd_pct:.1f}% >= {max_dd}%\n"
        f"Equity={equity:.2f}  Shutting down."
    )
    return send_alert(msg)


def send_startup_alert(symbol, model_name, risk, balance):
    """Send a bot started alert."""
    msg = (
        f"🚀 <b>LANE B LIVE TRADING STARTED</b>\n"
        f"Symbol={symbol}  Model={model_name}\n"
        f"Risk={risk:.1f}%  Balance={balance:.2f}"
    )
    return send_alert(msg)


def send_shutdown_alert(symbol, balance, equity, trades_taken=0):
    """Send a bot stopped alert."""
    pnl = equity - balance
    sign = "+" if pnl >= 0 else ""
    msg = (
        f"🛑 <b>LANE B STOPPED</b>\n"
        f"Symbol={symbol}  Trades={trades_taken}\n"
        f"PnL={sign}{pnl:.2f}  Equity={equity:.2f}"
    )
    return send_alert(msg)
