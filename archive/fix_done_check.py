import sys
path = r'C:\supreme-chainsaw\training\pretrain_lstm.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Change `if done:` to `if np.any(done):` to handle vectorized env returns
old_done = (
    '        obs, _, done, _ = env.step(neutral)\n'
    '        if done:\n'
    '            obs = env.reset()'
)
new_done = (
    '        obs, _, done, _ = env.step(neutral)\n'
    '        if np.any(done):\n'
    '            obs = env.reset()'
)
if old_done in content:
    content = content.replace(old_done, new_done, 1)
    print('OK: Fixed `if done:` to `if np.any(done):`')
else:
    print('WARN: done pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: pretrain_lstm.py updated')
