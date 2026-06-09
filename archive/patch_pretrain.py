#!/usr/bin/env python3
"""Add LSTM pretraining call into _train_once() before model.learn()."""
import re

path = r"C:\supreme-chainsaw\training\train_drl.py"

with open(path, "r") as f:
    content = f.read()

# Insert pretraining before model.learn()
old_learn = ('    logger.info("Starting PPO training")\n'
             '    try:\n'
             '        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)')

new_learn = ('    # LSTM pretraining: give the feature extractor useful initial representations\n'
             '    try:\n'
             '        from training.pretrain_lstm import pretrain_feature_extractor\n'
             '        pretrain_steps = int(os.environ.get("AGI_LSTM_PRETRAIN_STEPS", "200"))\n'
             '        if pretrain_steps > 0:\n'
             '            logger.info(f"Starting LSTM autoencoder pretrain ({pretrain_steps} gradient steps)...\\n"\n'
             '                         f"  This warms up the LSTM to recognize bar patterns before PPO training.\\n"\n'
             '                         f"  Disable with AGI_LSTM_PRETRAIN_STEPS=0")\n'
             '            pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
             '    except Exception as pre_err:\n'
             '        logger.warning(f"LSTM pretrain skipped (non-fatal): {pre_err}")\n'
             '\n'
             '    logger.info("Starting PPO training")\n'
             '    try:\n'
             '        model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)')

if old_learn in content:
    content = content.replace(old_learn, new_learn, 1)
    print("OK: Added pretraining call before model.learn()")
else:
    print("WARN: Could not find exact model.learn() location")
    # Try fuzzy match
    if 'model.learn(total_timesteps=total_timesteps, callback=callbacks' in content:
        content = content.replace(
            "logger.info(\"Starting PPO training\")",
            '    # LSTM pretrain: warm up feature extractor representations\n'
            '    try:\n'
            '        from training.pretrain_lstm import pretrain_feature_extractor\n'
            '        pretrain_steps = int(os.environ.get("AGI_LSTM_PRETRAIN_STEPS", "200"))\n'
            '        if pretrain_steps > 0:\n'
            '            logger.info(f"LSTM autoencoder pretrain ({pretrain_steps} steps)... Disable: AGI_LSTM_PRETRAIN_STEPS=0")\n'
            '            pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
            '    except Exception as pre_err:\n'
            '        logger.warning(f"LSTM pretrain skipped: {pre_err}")\n'
            '\n'
            '    logger.info("Starting PPO training")',
            1
        )
        print("OK: Added pretraining via fuzzy match")

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
