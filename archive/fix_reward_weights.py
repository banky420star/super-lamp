import sys

# ====== 1. trading_env.py: reward_weights + bonus_contrib ======
path1 = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path1, 'r', encoding='utf-8') as f:
    c1 = f.read()

# Change growth: 8.0 → 40.0
c1 = c1.replace('\"growth\": float(w.get(\"growth\", 8.0))', '\"growth\": float(w.get(\"growth\", 40.0))')

# Change payoff: 2.0 → 5.0
c1 = c1.replace('\"payoff\": float(w.get(\"payoff\", 2.0))', '\"payoff\": float(w.get(\"payoff\", 5.0))')

# Change drawdown_penalty: 1.0 → 2.0
c1 = c1.replace('\"drawdown_penalty\": float(w.get(\"drawdown_penalty\", 1.0))', '\"drawdown_penalty\": float(w.get(\"drawdown_penalty\", 2.0))')

# Change churn_penalty: 0.5 → 0.02
c1 = c1.replace('\"churn_penalty\": float(w.get(\"churn_penalty\", 0.5))', '\"churn_penalty\": float(w.get(\"churn_penalty\", 0.02))')

# Add quadratic loss amplification term to bonus_contrib
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
    '        # Quadratic loss amplification: bigger losses get disproportionately larger penalties\n'
    '        # e.g., -0.1% return → -0.04, -2% return → -16.0 (quadratic, not linear)\n'
    '        quad_loss = rw[\"growth\"] * (step_ret ** 2) * 100.0 if step_ret < 0 else 0.0\n'
    '        bonus_contrib = (\n'
    '            rw[\"growth\"] * growth_term\n'
    '            + rw[\"payoff\"] * payoff\n'
    '            + rw[\"sharpe_bonus\"] * sharpe_bonus\n'
    '            + rw[\"directional_followthrough\"] * directional_followthrough\n'
    '            + rw[\"actionable_target_bonus\"] * actionable_target_bonus\n'
    '            - quad_loss\n'
    '        )'
)
if old_bonus in c1:
    c1 = c1.replace(old_bonus, new_bonus, 1)
    print('OK: Added quadratic loss term to bonus_contrib')
else:
    print('WARN: bonus_contrib pattern not found')

with open(path1, 'w', encoding='utf-8') as f:
    f.write(c1)
print('OK: trading_env.py reward weights updated')

# ====== 2. reward_function.py: overtrading_penalty_coeff ======
path2 = r'C:\supreme-chainsaw\Python\rewards\reward_function.py'
with open(path2, 'r', encoding='utf-8') as f:
    c2 = f.read()

# Change overtrading_penalty_coeff: 0.5 → 0.02
c2 = c2.replace('overtrading_penalty_coeff: float = 0.5,', 'overtrading_penalty_coeff: float = 0.02,')

# Also update self assignment
c2 = c2.replace(
    'self.overtrading_penalty_coeff = float(overtrading_penalty_coeff)',
    'self.overtrading_penalty_coeff = float(overtrading_penalty_coeff)  # lowered 0.5->0.02: churn is expense, not punishment'
)

with open(path2, 'w', encoding='utf-8') as f:
    f.write(c2)
print('OK: reward_function.py overtrading_penalty_coeff updated')

print('DONE')
