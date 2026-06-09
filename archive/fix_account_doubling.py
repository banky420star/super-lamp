import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix 1: Add _last_equity_multiple to reset()
old_reset = (
    "        self._combo_multiplier: float = 1.0  # combo meter: each win +0.5x up to 5.0x, resets on loss\n"
    "        # Reset DSR"
)
new_reset = (
    "        self._combo_multiplier: float = 1.0  # combo meter: each win +0.5x up to 5.0x, resets on loss\n"
    "        self._last_equity_multiple: float = 1.0  # account doubling tracker: 1x, 2x, 4x, 8x...\n"
    "        # Reset DSR"
)
if old_reset in c:
    c = c.replace(old_reset, new_reset, 1)
    print('OK: Added _last_equity_multiple to reset()')
else:
    print('WARN: reset pattern not found')

# Fix 2: Add account-doubling check after equity update, before step_ret
old_equity = (
    "        self.peak_equity = max(self.peak_equity, self.equity)\n"
    "        self.equity_curve.append(float(self.equity))\n"
    "        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-12)"
)
new_equity = (
    "        self.peak_equity = max(self.peak_equity, self.equity)\n"
    "        self.equity_curve.append(float(self.equity))\n"
    "        # Account doubling bonus: massive reward every time equity doubles\n"
    "        curr_multiple = self.equity / max(self.initial_balance, 1.0)\n"
    "        account_doubling_bonus = 0.0\n"
    "        while curr_multiple >= self._last_equity_multiple * 2.0:\n"
    "            self._last_equity_multiple *= 2.0\n"
    "            account_doubling_bonus += self._last_equity_multiple * 10.0  # 2x=+20, 4x=+40, 8x=+80...\n"
    "        drawdown = (self.peak_equity - self.equity) / (self.peak_equity + 1e-12)"
)
if old_equity in c:
    c = c.replace(old_equity, new_equity, 1)
    print('OK: Added account-doubling bonus check')
else:
    print('WARN: equity pattern not found')

# Fix 3: Add account_doubling_bonus to bonus_contrib
old_bonus = (
    "            + win_streak_bonus\n"
    "        ) * combo_multiplier  # combo meter amplifies all bonuses"
)
new_bonus = (
    "            + win_streak_bonus\n"
    "            + account_doubling_bonus\n"
    "        ) * combo_multiplier  # combo meter amplifies all bonuses"
)
if old_bonus in c:
    c = c.replace(old_bonus, new_bonus, 1)
    print('OK: Added account_doubling_bonus to bonus_contrib')
else:
    print('WARN: bonus pattern not found')

# Fix 4: Add account_doubling_bonus to reward accumulation for logging
old_ra = (
    "            ra[\"position\"] = ra.get(\"position\", 0.0) + abs(self.position)\n"
    "            ra[\"delta\"] = ra.get(\"delta\", 0.0) + abs(delta)"
)
new_ra = (
    "            ra[\"position\"] = ra.get(\"position\", 0.0) + abs(self.position)\n"
    "            ra[\"delta\"] = ra.get(\"delta\", 0.0) + abs(delta)\n"
    "            ra[\"account_doubling_bonus\"] = ra.get(\"account_doubling_bonus\", 0.0) + account_doubling_bonus"
)
if old_ra in c:
    c = c.replace(old_ra, new_ra, 1)
    print('OK: Added account_doubling_bonus to reward accumulator')
else:
    print('WARN: ra pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
