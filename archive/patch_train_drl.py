#!/usr/bin/env python3
"""Fix train_drl.py: clean up the n_envs memory warning insertion."""
import re

path = r"C:\supreme-chainsaw\training\train_drl.py"

with open(path, "r") as f:
    content = f.read()

# Remove the broken memory warning block and reinsert correctly
# First find and remove the broken block
old_block = """    n_envs = int(os.environ.get("AGI_N_ENVS", "4"))
    # Memory warning: estimate rollout buffer size
    try:
        _obs_dim = int(df_pd.shape[1]) if hasattr(df_pd, 'shape') else 0
        if _obs_dim < 100:
            from Python.feature_pipeline import feature_count_for_version
            _n_feat = feature_count_for_version(feature_version)
            _obs_dim = 100 * _n_feat + 3
        _buf_mb = (n_envs * n_steps * _obs_dim * 4) / (1024 * 1024)
        if _buf_mb > 800:
            logger.warning(
                f"MEMORY: rollout buffer ~{_buf_mb:.0f}MB ({n_envs} envs x {n_steps} steps x {_obs_dim} dims). "
                "If OOM occurs, lower AGI_N_ENVS or AGI_PPO_N_STEPS."
            )
    except Exception:
        pass"""

new_block = """    n_envs = int(os.environ.get("AGI_N_ENVS", "4"))
    # Memory warning: estimate rollout buffer size for stability (V5 FIX)
    try:
        _obs_dim_est = int(df_pd.shape[1]) if hasattr(df_pd, 'shape') and df_pd.shape[1] > 100 else 16403
        _buf_mb_est = (n_envs * n_steps * _obs_dim_est * 4) / (1024 * 1024)
        if _buf_mb_est > 800:
            logger.warning(
                f"MEMORY: rollout buffer ~{_buf_mb_est:.0f}MB ({n_envs} envs x {n_steps} steps x {_obs_dim_est} dims). "
                "If OOM occurs, lower AGI_N_ENVS or AGI_PPO_N_STEPS."
            )
    except Exception:
        pass"""

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print("OK: Fixed memory warning block")
else:
    # Check if the broken block is not there - might need to insert it fresh
    # Look for the clean line
    clean_line = '    n_envs = int(os.environ.get("AGI_N_ENVS", "4"))'
    if clean_line in content:
        # Check if the memory warning block is already there correctly
        if "Memory warning" not in content:
            content = content.replace(clean_line, new_block, 1)
            print("OK: Inserted fresh memory warning block")
        else:
            print("Memory warning already present - checking indentation")
    else:
        print("ERR: Could not find n_envs line")

try:
    compile(content, path, "exec")
    with open(path, "w") as f:
        f.write(content)
    print(f"OK: Saved {path}")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    lines = content.split("\n")
    if e.lineno:
        for j in range(max(0, e.lineno - 5), min(len(lines), e.lineno + 3)):
            marker = ">>>" if j == e.lineno - 1 else "   "
            print(f"  {marker} {j+1}: {lines[j]}")
