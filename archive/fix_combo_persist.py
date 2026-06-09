import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Fix: Read combo multiplier from persistent self._combo_multiplier, not ephemeral _trade_metrics
old = (
    "        combo_multiplier = self._trade_metrics.get(\"combo_multiplier\", 1.0)"
)
new = (
    "        combo_multiplier = getattr(self, \"_combo_multiplier\", 1.0)  # persistent across steps, only reset on loss"
)
if old in c:
    c = c.replace(old, new, 1)
    print('OK: Fixed combo_multiplier to read from self._combo_multiplier')
else:
    print('WARN: pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)
print('OK: trading_env.py updated')
