#!/usr/bin/env python3
"""Patch _policy_kwargs_for() in train_drl.py to pass regime_dim."""
import re

path = r"C:\supreme-chainsaw\training\train_drl.py"

with open(path, "r") as f:
    content = f.read()

# Fix 1: Add import for NUM_REGIMES in _policy_kwargs_for
old_imports = "from drl.lstm_feature_extractor import LSTMFeatureExtractor"
new_imports = (
    "from drl.lstm_feature_extractor import LSTMFeatureExtractor\n"
    "from drl.regime_detector import NUM_REGIMES"
)
if old_imports in content:
    content = content.replace(old_imports, new_imports, 1)
    print("OK: Added NUM_REGIMES import")
else:
    print("WARN: Could not find import line")

# Fix 2: Add regime_dim to features_extractor_kwargs
old_kwargs = (
    "            features_extractor_kwargs=dict(features_dim=256, window_size=100, num_heads=_attn_heads),"
)
# Check if regime is enabled and add regime_dim
new_kwargs_1 = (
    "            features_extractor_kwargs=dict(features_dim=256, window_size=100, num_heads=_attn_heads,\n"
    "                                           regime_dim=NUM_REGIMES + 1 if os.environ.get('AGI_USE_REGIME', '0') == '1' else 0),"
)
if old_kwargs in content:
    content = content.replace(old_kwargs, new_kwargs_1, 1)
    print("OK: Added regime_dim to features_extractor_kwargs")
else:
    print("WARN: Could not find kwargs line 1")

# Also handle the non-ULTIMATE_150 case (ENGINEERED_V2 uses LSTMFeatureExtractor)
# The LSTMFeatureExtractor also needs the regime_dim treatment, but it's a different class
# For now leave ENGINEERED_V2 unchanged since it's the legacy path

# Fix 3: Add memory usage warning in _train_once
# Find the line where n_envs is set
old_n_envs = "    n_envs = int(os.environ.get(\"AGI_N_ENVS\", \"4\"))"
new_n_envs = """    n_envs = int(os.environ.get("AGI_N_ENVS", "4"))
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

if old_n_envs in content:
    content = content.replace(old_n_envs, new_n_envs, 1)
    print("OK: Added memory warning after n_envs")
else:
    print("WARN: Could not find n_envs line")

# Verify
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
