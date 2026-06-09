import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix 1: Correct the comment math
old_comment = (
    '        # Quadratic loss amplification: bigger losses get disproportionately larger penalties\n'
    '        # e.g., -0.1% return → -0.04, -2% return → -16.0 (quadratic, not linear)\n'
    '        quad_loss = rw[\"growth\"] * (step_ret ** 2) * 100.0 if step_ret < 0 else 0.0'
)
new_comment = (
    '        # Quadratic loss amplification: bigger losses get disproportionately larger penalties\n'
    '        # e.g., -0.1% return → -0.004, -2% return → -1.6 (vs linear -0.8)\n'
    '        quad_loss = rw[\"growth\"] * (step_ret ** 2) * 100.0 if step_ret < 0 else 0.0'
)
if old_comment in c:
    c = c.replace(old_comment, new_comment, 1)
    print('OK: Fixed comment math')
else:
    print('WARN: comment pattern not found')

# Fix 2: Move quad_loss from bonus_contrib to penalty_contrib
old_bonus = (
    '        bonus_contrib = (\n'
    '            rw[\"growth\"] * growth_term\n'
    '            + rw[\"payoff\"] * payoff\n'
    '            + rw[\"sharpe_bonus\"] * sharpe_bonus\n'
    '            + rw[\"directional_followthrough\"] * directional_followthrough\n'
    '            + rw[\"actionable_target_bonus\"] * actionable_target_bonus\n'
    '            - quad_loss\n'
    '        )'
)
new_bonus = (
    '        bonus_contrib = (\n'
    '            rw[\"growth\"] * growth_term\n'
    '            + rw[\"payoff\"] * payoff\n'
    '            + rw[\"sharpe_bonus\"] * sharpe_bonus\n'
    '            + rw[\"directional_followthrough\"] * directional_followthrough\n'
    '            + rw[\"actionable_target_bonus\"] * actionable_target_bonus\n'
    '        )'
)
if old_bonus in c:
    c = c.replace(old_bonus, new_bonus, 1)
    print('OK: Removed quad_loss from bonus_contrib')
else:
    print('WARN: bonus_contrib pattern not found')

# Fix 3: Add quad_loss to penalty_contrib
old_penalty = (
    '        penalty_contrib = (\n'
    '            rw[\"drawdown_penalty\"] * dd_base\n'
    '            + rw[\"cost_penalty\"] * cost_penalty\n'
    '            + rw[\"churn_penalty\"] * churn_penalty\n'
    '            + loss_streak_penalty\n'
    '            + rw[\"neutral_collapse_penalty\"] * neutral_collapse_penalty\n'
    '        )'
)
new_penalty = (
    '        penalty_contrib = (\n'
    '            rw[\"drawdown_penalty\"] * dd_base\n'
    '            + rw[\"cost_penalty\"] * cost_penalty\n'
    '            + rw[\"churn_penalty\"] * churn_penalty\n'
    '            + loss_streak_penalty\n'
    '            + rw[\"neutral_collapse_penalty\"] * neutral_collapse_penalty\n'
    '            + quad_loss  # quadratic loss amplification (scaled by penalty_scale below)\n'
    '        )'
)
if old_penalty in c:
    c = c.replace(old_penalty, new_penalty, 1)
    print('OK: Added quad_loss to penalty_contrib')
else:
    print('WARN: penalty_contrib pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
