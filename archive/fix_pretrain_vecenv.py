import sys
path = r'C:\supreme-chainsaw\training\pretrain_lstm.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The issue: with VecEnv (n_envs > 1), obs.shape is [n_envs, obs_dim]
# The code stacks these as [n_timesteps, n_envs, obs_dim], then tries to 
# view(batch_size, 100, -1) which fails dimensionally.
# Fix: flatten observations to [n_envs * obs_dim] when they have >1 dim

old = (
    "    obs = env.reset()\n"
    "    for _ in range(max(steps * 2, 200)):\n"
    "        all_obs.append(torch.from_numpy(obs.copy()).float())\n"
    "        obs, _, done, _ = env.step(neutral)\n"
    "        if np.any(done):\n"
    "            obs = env.reset()"
)
new = (
    "    obs = env.reset()\n"
    "    for _ in range(max(steps * 2, 200)):\n"
    "        obs_flat = obs.reshape(-1) if obs.ndim > 1 else obs\n"
    "        all_obs.append(torch.from_numpy(obs_flat.copy()).float())\n"
    "        obs, _, done, _ = env.step(neutral)\n"
    "        if np.any(done):\n"
    "            obs = env.reset()"
)
if old in content:
    content = content.replace(old, new, 1)
    print('OK: Added obs flattening for VecEnv')
else:
    print('WARN: Pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: pretrain_lstm.py updated')
