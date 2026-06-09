import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix 1: Replace _win_streak with _combo_multiplier in reset()
old_reset = (
    "        self._win_streak: int = 0  # consecutive wins, resets on loss\n"
    "        # Reset DSR"
)
new_reset = (
    "        self._combo_multiplier: float = 1.0  # combo meter: each win +0.5x up to 5.0x, resets on loss\n"
    "        # Reset DSR"
)
if old_reset in c:
    c = c.replace(old_reset, new_reset, 1)
    print('OK: Replaced _win_streak with _combo_multiplier in reset()')
else:
    print('WARN: reset pattern not found')

# Fix 2: Replace win_streak_bonus logic with combo multiplier in _close_trade()
old_streak = (
    "        # Win streak: consecutive profitable trades = escalating bonus\n"
    "        # 5 wins >$10 = huge reward, 10 wins = massive, resets on any loss\n"
    "        if total_pnl > 10.0:\n"
    "            self._win_streak += 1\n"
    "        else:\n"
    "            self._win_streak = 0\n"
    "        # Bonus escalates with streak: 1=0, 2=0.2, 3=0.5, 5=2.0, 10=5.0\n"
    "        if self._win_streak >= 10:\n"
    "            win_streak_bonus = 5.0\n"
    "        elif self._win_streak >= 5:\n"
    "            win_streak_bonus = 2.0\n"
    "        elif self._win_streak >= 3:\n"
    "            win_streak_bonus = 0.5\n"
    "        elif self._win_streak >= 2:\n"
    "            win_streak_bonus = 0.2\n"
    "        else:\n"
    "            win_streak_bonus = 0.0"
)
new_streak = (
    "        # Combo meter: each consecutive profitable trade amplifies ALL rewards\n"
    "        # +0.5x per win, capped at 5.0x. Resets to 1.0x on any loss or sub-$10 win\n"
    "        if total_pnl > 10.0:\n"
    "            self._combo_multiplier = min(5.0, self._combo_multiplier + 0.5)\n"
    "        else:\n"
    "            self._combo_multiplier = 1.0"
)
if old_streak in c:
    c = c.replace(old_streak, new_streak, 1)
    print('OK: Replaced win_streak with combo multiplier in _close_trade()')
else:
    print('WARN: streak pattern not found')

# Fix 3: Replace win_streak_bonus metrics entries with combo_multiplier
old_metrics = (
    '            \"win_streak_bonus\": win_streak_bonus,\n'
    '            \"win_streak\": self._win_streak,'
)
new_metrics = (
    '            \"combo_multiplier\": self._combo_multiplier,'
)
if old_metrics in c:
    c = c.replace(old_metrics, new_metrics, 1)
    print('OK: Replaced win_streak metrics with combo_multiplier')
else:
    print('WARN: metrics pattern not found')

# Fix 4: Replace win_streak_bonus extraction in step with combo_multiplier
old_extract = (
    "        win_streak_bonus = self._trade_metrics.get(\"win_streak_bonus\", 0.0)\n"
    "        exit_timing_bonus = self._trade_metrics.get(\"exit_timing_bonus\", 0.0)"
)
new_extract = (
    "        exit_timing_bonus = self._trade_metrics.get(\"exit_timing_bonus\", 0.0)\n"
    "        combo_multiplier = self._trade_metrics.get(\"combo_multiplier\", 1.0)"
)
if old_extract in c:
    c = c.replace(old_extract, new_extract, 1)
    print('OK: Replaced win_streak_bonus extraction with combo_multiplier')
else:
    print('WARN: extract pattern not found')

# Fix 5: Replace + win_streak_bonus in bonus_contrib with * combo_multiplier at the end
old_bonus = (
    "            + close_bonus\n"
    "            + win_streak_bonus\n"
    "            + exit_timing_bonus\n"
    "        )"
)
new_bonus = (
    "            + close_bonus\n"
    "            + exit_timing_bonus\n"
    "        ) * combo_multiplier  # combo meter amplifies all bonuses"
)
if old_bonus in c:
    c = c.replace(old_bonus, new_bonus, 1)
    print('OK: Replaced win_streak bonus with combo multiplier in bonus_contrib')
else:
    print('WARN: bonus pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
