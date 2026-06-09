import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix 1: Make exit_quality_reward proportional to profit magnitude
old_close = (
    "        exit_quality_reward = 1.0 if exit_type == \"tp\" else -1.0"
)
new_close = (
    "        # Proportional close bonus: bigger profit = bigger reward, bigger loss = bigger penalty\n"
    "        profit_pct = profit / (entry_price + 1e-12)\n"
    "        exit_quality_reward = profit_pct * 50.0  # e.g. +1% TP = +0.5, -2% SL = -1.0"
)
if old_close in c:
    c = c.replace(old_close, new_close, 1)
    print('OK: Made exit_quality_reward proportional to profit')
else:
    print('WARN: exit_quality_reward pattern not found')

# Fix 2: Add close_bonus to bonus_contrib
old_bonus = (
    '        bonus_contrib = (\n'
    '            rw[\"growth\"] * growth_term\n'
    '            + rw[\"payoff\"] * payoff\n'
    '            + rw[\"sharpe_bonus\"] * sharpe_bonus\n'
    '            + rw[\"directional_followthrough\"] * directional_followthrough\n'
    '            + rw[\"actionable_target_bonus\"] * actionable_target_bonus\n'
    '        )'
)
new_bonus = (
    '        # Close bonus: every closed trade in profit gives extra reward proportional to profit\n'
    '        close_bonus = self._trade_metrics.get(\"exit_quality_reward\", 0.0)\n'
    '        bonus_contrib = (\n'
    '            rw[\"growth\"] * growth_term\n'
    '            + rw[\"payoff\"] * payoff\n'
    '            + rw[\"sharpe_bonus\"] * sharpe_bonus\n'
    '            + rw[\"directional_followthrough\"] * directional_followthrough\n'
    '            + rw[\"actionable_target_bonus\"] * actionable_target_bonus\n'
    '            + close_bonus\n'
    '        )'
)
if old_bonus in c:
    c = c.replace(old_bonus, new_bonus, 1)
    print('OK: Added close_bonus to bonus_contrib')
else:
    print('WARN: bonus_contrib pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
