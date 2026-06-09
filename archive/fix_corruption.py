import sys
path = r'C:\supreme-chainsaw\drl\trading_env.py'
with open(path, 'r', encoding='utf-8') as f:
    c = f.read()

# Find and fix the corrupted line
old = '_doublings = 0        while curr_multiple >= self._last_equity_multiple * 2.0 and _doublings < 10:  # safety: _doublings cap prevents infinite loops'
new = '_doublings = 0\n        while curr_multiple >= self._last_equity_multiple * 2.0 and _doublings < 10:  # safety cap'
if old in c:
    c = c.replace(old, new, 1)
    print('OK: Fixed corrupted line')
else:
    # Try just finding the substring
    idx = c.find('_doublings = 0        while')
    if idx >= 0:
        prefix = c[:idx]
        # Find where this corrupted block ends
        end_marker = '  # safety: _doublings cap prevents infinite loops'
        end_idx = c.find(end_marker, idx)
        if end_idx >= 0:
            rest = c[end_idx + len(end_marker):]
            c = prefix + '_doublings = 0\n        while curr_multiple >= self._last_equity_multiple * 2.0 and _doublings < 10:  # safety cap' + rest
            print('OK: Fixed corrupted line (manual)')
        else:
            print('FAIL: Could not find end marker')
    else:
        print('Source looks clean already')

with open(path, 'w', encoding='utf-8') as f:
    f.write(c)

# Verify
import drl.trading_env as te
print('[OK] Module compiles')
