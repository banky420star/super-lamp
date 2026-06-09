import sys
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The pretrain block is now before callback construction. When the except block runs,
# pretrain_loss_reduction is never defined. We need to initialize it before the try block.

old_block = (
    '    # LSTM pretraining: give the feature extractor useful initial representations\n'
    '    try:\n'
    '        from training.pretrain_lstm import pretrain_feature_extractor\n'
    '        pretrain_steps = int(os.environ.get("AGI_LSTM_PRETRAIN_STEPS", "200"))\n'
    '        if pretrain_steps > 0:\n'
    '            logger.info(f"Starting LSTM autoencoder pretrain ({pretrain_steps} gradient steps)...\\n"\n'
    '                         f"  This warms up the LSTM to recognize bar patterns before PPO training.\\n"\n'
    '                         f"  Disable with AGI_LSTM_PRETRAIN_STEPS=0")\n'
    '            pretrain_result = pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
    '            pretrain_loss_reduction = pretrain_result[1] if isinstance(pretrain_result, tuple) and pretrain_result[0] else 0.0\n'
    '            # Reset env after pretraining to avoid state pollution into PPO\n'
    '            try:\n'
    '                env.reset()\n'
    '            except Exception:\n'
    '                pass\n'
    '    except Exception as pre_err:\n'
    '        logger.warning(f"LSTM pretrain skipped (non-fatal): {pre_err}")\n'
)

new_block = (
    '    # LSTM pretraining: give the feature extractor useful initial representations\n'
    '    pretrain_loss_reduction = 0.0\n'
    '    try:\n'
    '        from training.pretrain_lstm import pretrain_feature_extractor\n'
    '        pretrain_steps = int(os.environ.get("AGI_LSTM_PRETRAIN_STEPS", "200"))\n'
    '        if pretrain_steps > 0:\n'
    '            logger.info(f"Starting LSTM autoencoder pretrain ({pretrain_steps} gradient steps)...\\n"\n'
    '                         f"  This warms up the LSTM to recognize bar patterns before PPO training.\\n"\n'
    '                         f"  Disable with AGI_LSTM_PRETRAIN_STEPS=0")\n'
    '            pretrain_result = pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
    '            pretrain_loss_reduction = pretrain_result[1] if isinstance(pretrain_result, tuple) and pretrain_result[0] else 0.0\n'
    '            # Reset env after pretraining to avoid state pollution into PPO\n'
    '            try:\n'
    '                env.reset()\n'
    '            except Exception:\n'
    '                pass\n'
    '    except Exception as pre_err:\n'
    '        logger.warning(f"LSTM pretrain skipped (non-fatal): {pre_err}")\n'
)

if old_block in content:
    content = content.replace(old_block, new_block, 1)
    print('OK: Added pretrain_loss_reduction = 0.0 before try block')
else:
    print('WARN: Could not find pretrain block')
    # Try with single backslashes
    idx = content.find('pretrain_steps = int(os.environ.get("AGI_LSTM_PRETRAIN_STEPS", "200"))')
    if idx >= 0:
        print(f'  Found pretrain_steps line at {idx}')
        # Look backwards for '# LSTM pretraining'
        marker = content.rfind('# LSTM pretraining', 0, idx)
        if marker >= 0:
            print(f'  Found marker at {marker}')
            print(f'  Context: {repr(content[marker:marker+100])}')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: train_drl.py updated')
