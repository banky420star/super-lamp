import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix 1: Add exit_timing_bonus calculation after trailing_efficiency
old_timing = (
    '        trailing_efficiency = min(1.0, abs(profit) / base) if max_fav > 0 else 0.0\n'
    '        trailing_moves = int(self.open_trade.get(\"trailing_moves\", 0))\n'
    '        if exit_type == \"sl\" and profit < 0:'
)
new_timing = (
    '        trailing_efficiency = min(1.0, abs(profit) / base) if max_fav > 0 else 0.0\n'
    '        trailing_moves = int(self.open_trade.get(\"trailing_moves\", 0))\n'
    '        # Exit timing bonus: closing near the candle high (long) or low (short) = good timing\n'
    '        candle_high = float(self.highs[self.current_step])\n'
    '        candle_low = float(self.lows[self.current_step])\n'
    '        candle_range = max(candle_high - candle_low, 1e-8)\n'
    '        if direction > 0:\n'
    '            # Long: reward exits near candle high\n'
    '            exit_timing = 1.0 - (candle_high - exit_price) / candle_range\n'
    '        else:\n'
    '            # Short: reward exits near candle low\n'
    '            exit_timing = 1.0 - (exit_price - candle_low) / candle_range\n'
    '        exit_timing = float(np.clip(exit_timing, 0.0, 1.0))\n'
    '        exit_timing_bonus = exit_timing * 0.3  # up to +0.3 for perfect timing\n'
    '        if exit_type == \"sl\" and profit < 0:'
)
if old_timing in c:
    c = c.replace(old_timing, new_timing, 1)
    print('OK: Added exit_timing_bonus calculation')
else:
    print('WARN: timing pattern not found')

# Fix 2: Add exit_timing_bonus to _trade_metrics
old_metrics_timing = (
    '            \"win_streak_bonus\": win_streak_bonus,\n'
    '            \"win_streak\": self._win_streak,'
)
new_metrics_timing = (
    '            \"win_streak_bonus\": win_streak_bonus,\n'
    '            \"win_streak\": self._win_streak,\n'
    '            \"exit_timing_bonus\": exit_timing_bonus,\n'
    '            \"exit_timing\": exit_timing,'
)
if old_metrics_timing in c:
    c = c.replace(old_metrics_timing, new_metrics_timing, 1)
    print('OK: Added exit_timing_bonus to _trade_metrics')
else:
    print('WARN: metrics_timing pattern not found')

# Fix 3: Add exit_timing_bonus extraction in reward calc (after win_streak_bonus line)
old_extract = (
    '        win_streak_bonus = self._trade_metrics.get(\"win_streak_bonus\", 0.0)\n'
    '        bonus_contrib = ('
)
new_extract = (
    '        win_streak_bonus = self._trade_metrics.get(\"win_streak_bonus\", 0.0)\n'
    '        exit_timing_bonus = self._trade_metrics.get(\"exit_timing_bonus\", 0.0)\n'
    '        bonus_contrib = ('
)
if old_extract in c:
    c = c.replace(old_extract, new_extract, 1)
    print('OK: Added exit_timing_bonus extraction in reward calc')
else:
    print('WARN: extract pattern not found')

# Fix 4: Add exit_timing_bonus to bonus_contrib tuple
old_bonus_timing = (
    '            + win_streak_bonus\n'
    '        )'
)
new_bonus_timing = (
    '            + win_streak_bonus\n'
    '            + exit_timing_bonus\n'
    '        )'
)
if old_bonus_timing in c:
    c = c.replace(old_bonus_timing, new_bonus_timing, 1)
    print('OK: Added exit_timing_bonus to bonus_contrib tuple')
else:
    print('WARN: bonus_timing pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
