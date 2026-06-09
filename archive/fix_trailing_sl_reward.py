import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Replace the exit_quality_reward calculation with trailing-aware logic
old = (
    "        # Proportional close bonus: bigger profit = bigger reward, bigger loss = bigger penalty\n"
    "        profit_pct = profit / (entry_price + 1e-12)\n"
    "        exit_quality_reward = profit_pct * 50.0  # e.g. +1% TP = +0.5, -2% SL = -1.0"
)
new = (
    "        # Close bonus: proportional to profit, with special cases for trailing SL and SL losses\n"
    "        profit_pct = profit / (entry_price + 1e-12)\n"
    "        trailing_moves = int(self.open_trade.get(\"trailing_moves\", 0))\n"
    "        if exit_type == \"sl\" and profit < 0:\n"
    "            # Stop loss hit: low flat negative (cost of trading, not a major penalty)\n"
    "            exit_quality_reward = -0.15\n"
    "            trailing_bonus = 0.0\n"
    "        elif exit_type == \"sl\" and profit > 0 and trailing_moves > 0:\n"
    "            # Trailing SL hit in profit: reward good trailing management\n"
    "            base_reward = profit_pct * 50.0\n"
    "            trail_bonus = trailing_efficiency * 0.5  # up to +0.5 for perfect trailing\n"
    "            exit_quality_reward = base_reward + trail_bonus\n"
    "        else:\n"
    "            # TP or time_exit: proportional to profit/loss magnitude\n"
    "            exit_quality_reward = profit_pct * 50.0\n"
    "            trailing_bonus = 0.0"
)
if old in c:
    c = c.replace(old, new, 1)
    print('OK: Updated exit_quality_reward with trailing SL bonus and low SL loss penalty')
else:
    print('WARN: old pattern not found!')
    # Debug: find what's actually there
    idx = c.find('exit_quality_reward = profit_pct * 50.0')
    if idx >= 0:
        print(f'  Found at {idx}, context: {repr(c[idx-100:idx+80])}')

# Also add trailing_bonus to _trade_metrics dict so it flows through
old_metrics = (
    "            \"exit_quality_reward\": exit_quality_reward,\n"
    "            \"trailing_efficiency\": float(trailing_efficiency),"
)
new_metrics = (
    "            \"exit_quality_reward\": exit_quality_reward,\n"
    "            \"trailing_bonus\": trailing_bonus,\n"
    "            \"trailing_efficiency\": float(trailing_efficiency),"
)
if old_metrics in c:
    c = c.replace(old_metrics, new_metrics, 1)
    print('OK: Added trailing_bonus to _trade_metrics')
else:
    print('WARN: metrics pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
