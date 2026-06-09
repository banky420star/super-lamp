import sys
path = r'C:\supreme-chainsaw\training\pretrain_lstm.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The flatten fix was wrong - it changes obs_dim from 16409 to 32818 (2*16409),
# which breaks the LSTM's expected input_size.
# Correct fix: iterate over individual environment observations

old = (
    "    obs = env.reset()\n"
    "    for _ in range(max(steps * 2, 200)):\n"
    "        obs_flat = obs.reshape(-1) if obs.ndim > 1 else obs\n"
    "        all_obs.append(torch.from_numpy(obs_flat.copy()).float())\n"
    "        obs, _, done, _ = env.step(neutral)\n"
    "        if np.any(done):\n"
    "            obs = env.reset()"
)
new = (
    "    obs = env.reset()\n"
    "    for _ in range(max(steps * 2, 200)):\n"
    "        if obs.ndim > 1:\n"
    "            # VecEnv: extract each env's observation individually\n"
    "            for env_idx in range(obs.shape[0]):\n"
    "                all_obs.append(torch.from_numpy(obs[env_idx].copy()).float())\n"
    "        else:\n"
    "            all_obs.append(torch.from_numpy(obs.copy()).float())\n"
    "        obs, _, done, _ = env.step(neutral)\n"
    "        if np.any(done):\n"
    "            obs = env.reset()"
)
if old in content:
    content = content.replace(old, new, 1)
    print('OK: Fixed VecEnv handling - iterate individual envs')
else:
    print('WARN: Pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: pretrain_lstm.py updated')
