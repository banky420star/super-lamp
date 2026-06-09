#!/usr/bin/env python3
"""Fix import safety for NUM_REGIMES and ensure env reset after pretraining."""
import re

path = r"C:\supreme-chainsaw\training\train_drl.py"

with open(path, "r") as f:
    content = f.read()

# Fix 1: Remove the direct NUM_REGIMES import and replace with safe import inside _policy_kwargs_for
old_import = ("from drl.regime_detector import NUM_REGIMES")
new_import = (
    "# Regime detector import (safe - guarded for optional dependency)\n"
    "try:\n"
    "    from drl.regime_detector import NUM_REGIMES\n"
    "    _HAS_REGIME_IMPORT = True\n"
    "except Exception:\n"
    "    NUM_REGIMES = 5\n"
    "    _HAS_REGIME_IMPORT = False"
)
if old_import in content:
    content = content.replace(old_import, new_import, 1)
    print("OK: Guarded NUM_REGIMES import")
else:
    print("WARN: Could not find NUM_REGIMES import")

# Fix 2: Add env.reset() after pretraining to avoid env state pollution
old_pretrain = (
    "            logger.info(f\"LSTM autoencoder pretrain ({pretrain_steps} steps)... Disable: AGI_LSTM_PRETRAIN_STEPS=0\")\n"
    "            pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode=\"ae\", logger_obj=logger)\n"
    "    except Exception as pre_err:\n"
    "        logger.warning(f\"LSTM pretrain skipped: {pre_err}\")"
)
# The second version (inserted by patch_pretrain.py)
old_pretrain_2 = (
    '            logger.info(f"Starting LSTM autoencoder pretrain ({pretrain_steps} gradient steps)...\\n"\n'
    '                         f"  This warms up the LSTM to recognize bar patterns before PPO training.\\n"\n'
    '                         f"  Disable with AGI_LSTM_PRETRAIN_STEPS=0")\n'
    '            pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
    '    except Exception as pre_err:\n'
    '        logger.warning(f"LSTM pretrain skipped (non-fatal): {pre_err}")'
)

new_pretrain = (
    '            logger.info(f"Starting LSTM autoencoder pretrain ({pretrain_steps} gradient steps)...\\n"\n'
    '                         f"  This warms up the LSTM to recognize bar patterns before PPO training.\\n"\n'
    '                         f"  Disable with AGI_LSTM_PRETRAIN_STEPS=0")\n'
    '            pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
    '            # Reset env after pretraining to avoid state pollution into PPO\n'
    '            try:\n'
    '                env.reset()\n'
    '            except Exception:\n'
    '                pass\n'
    '    except Exception as pre_err:\n'
    '        logger.warning(f"LSTM pretrain skipped (non-fatal): {pre_err}")'
)

if old_pretrain_2 in content:
    content = content.replace(old_pretrain_2, new_pretrain, 1)
    print("OK: Added env.reset() after pretraining")
else:
    print("WARN: Could not find pretrain block (checking first version)")
    if old_pretrain in content:
        content = content.replace(old_pretrain, new_pretrain, 1)
        print("OK: Added env.reset() after pretraining (first version)")

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
