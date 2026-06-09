import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix both bugs by replacing the entire section from profit_pct to trailing_efficiency
old_section = (
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
    "            trailing_bonus = 0.0\n"
    "        max_fav = float(self.open_trade.get(\"max_fav\", 0.0))\n"
    "        base = max(max_fav * entry_price, 1e-4)\n"
    "        trailing_efficiency = min(1.0, abs(profit) / base) if max_fav > 0 else 0.0"
)
new_section = (
    "        # Close bonus: proportional to profit, with special cases for trailing SL and SL losses\n"
    "        profit_pct = profit / (entry_price + 1e-12)\n"
    "        max_fav = float(self.open_trade.get(\"max_fav\", 0.0))\n"
    "        base = max(max_fav * entry_price, 1e-4)\n"
    "        trailing_efficiency = min(1.0, abs(profit) / base) if max_fav > 0 else 0.0\n"
    "        trailing_moves = int(self.open_trade.get(\"trailing_moves\", 0))\n"
    "        if exit_type == \"sl\" and profit < 0:\n"
    "            # Stop loss hit: low flat negative (cost of trading, not a major penalty)\n"
    "            exit_quality_reward = -0.15\n"
    "            trailing_bonus = 0.0\n"
    "        elif exit_type == \"sl\" and profit > 0 and trailing_moves > 0:\n"
    "            # Trailing SL hit in profit: reward good trailing management\n"
    "            base_reward = profit_pct * 50.0\n"
    "            trailing_bonus = trailing_efficiency * 0.5  # up to +0.5 for perfect trailing\n"
    "            exit_quality_reward = base_reward + trailing_bonus\n"
    "        else:\n"
    "            # TP or time_exit: proportional to profit/loss magnitude\n"
    "            exit_quality_reward = profit_pct * 50.0\n"
    "            trailing_bonus = 0.0"
)
if old_section in c:
    c = c.replace(old_section, new_section, 1)
    print('OK: Fixed forward reference and naming bugs')
else:
    print('WARN: section pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
