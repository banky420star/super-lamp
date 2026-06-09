import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix 1: Add _win_streak to reset()
old_reset = (
    '        self._ep_max_favorable: float = 0.0\n'
    '        self._ep_max_adverse: float = 0.0'
)
new_reset = (
    '        self._ep_max_favorable: float = 0.0\n'
    '        self._ep_max_adverse: float = 0.0\n'
    '        self._win_streak: int = 0  # consecutive wins, resets on loss'
)
if old_reset in c:
    c = c.replace(old_reset, new_reset, 1)
    print('OK: Added _win_streak to reset()')
else:
    print('WARN: reset pattern not found')

# Fix 2: Add win streak tracking in _close_trade() after win/loss tally
old_streak = (
    '        self._ep_max_favorable = max(self._ep_max_favorable, max_fav)\n'
    '        self._ep_max_adverse = max(self._ep_max_adverse, float(self.open_trade.get(\"max_adv\", 0.0)))\n'
    '        self._trade_metrics = {'
)
new_streak = (
    '        self._ep_max_favorable = max(self._ep_max_favorable, max_fav)\n'
    '        self._ep_max_adverse = max(self._ep_max_adverse, float(self.open_trade.get(\"max_adv\", 0.0)))\n'
    '        # Win streak: consecutive profitable trades = escalating bonus\n'
    '        # 5 wins >$10 = huge reward, 10 wins = massive, resets on any loss\n'
    '        if total_pnl > 10.0:\n'
    '            self._win_streak += 1\n'
    '        else:\n'
    '            self._win_streak = 0\n'
    '        # Bonus escalates with streak: 1=0, 2=0.2, 3=0.5, 5=2.0, 10=5.0\n'
    '        if self._win_streak >= 10:\n'
    '            win_streak_bonus = 5.0\n'
    '        elif self._win_streak >= 5:\n'
    '            win_streak_bonus = 2.0\n'
    '        elif self._win_streak >= 3:\n'
    '            win_streak_bonus = 0.5\n'
    '        elif self._win_streak >= 2:\n'
    '            win_streak_bonus = 0.2\n'
    '        else:\n'
    '            win_streak_bonus = 0.0\n'
    '        self._trade_metrics = {'
)
if old_streak in c:
    c = c.replace(old_streak, new_streak, 1)
    print('OK: Added win streak tracking to _close_trade()')
else:
    print('WARN: streak pattern not found')

# Fix 3: Add win_streak_bonus to _trade_metrics dict
old_metrics_entries = (
    '            \"trailing_bonus\": trailing_bonus,\n'
    '            \"trailing_efficiency\": float(trailing_efficiency),'
)
new_metrics_entries = (
    '            \"trailing_bonus\": trailing_bonus,\n'
    '            \"trailing_efficiency\": float(trailing_efficiency),\n'
    '            \"win_streak_bonus\": win_streak_bonus,\n'
    '            \"win_streak\": self._win_streak,'
)
if old_metrics_entries in c:
    c = c.replace(old_metrics_entries, new_metrics_entries, 1)
    print('OK: Added win_streak_bonus to _trade_metrics')
else:
    print('WARN: metrics entries pattern not found')

# Fix 4: Add win_streak_bonus to bonus_contrib (after close_bonus line)
old_bonus = (
    '        close_bonus = self._trade_metrics.get(\"exit_quality_reward\", 0.0)\n'
    '        bonus_contrib = ('
)
new_bonus = (
    '        close_bonus = self._trade_metrics.get(\"exit_quality_reward\", 0.0)\n'
    '        win_streak_bonus = self._trade_metrics.get(\"win_streak_bonus\", 0.0)\n'
    '        bonus_contrib = ('
)
if old_bonus in c:
    c = c.replace(old_bonus, new_bonus, 1)
    print('OK: Added win_streak_bonus extraction in reward calc')
else:
    print('WARN: bonus pattern not found')

# Fix 5: Add + win_streak_bonus to bonus_contrib tuple
old_bonus_tuple = (
    '            + close_bonus\n'
    '        )'
)
new_bonus_tuple = (
    '            + close_bonus\n'
    '            + win_streak_bonus\n'
    '        )'
)
if old_bonus_tuple in c:
    c = c.replace(old_bonus_tuple, new_bonus_tuple, 1)
    print('OK: Added win_streak_bonus to bonus_contrib tuple')
else:
    print('WARN: bonus tuple pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
