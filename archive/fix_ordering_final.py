import sys, re
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Step 1: Remove the dead init line we added
old_init = (
    "    pretrain_loss_reduction = 0.0  # will be updated by LSTM pretrain below if enabled\n"
    "    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)"
)
new_no_init = (
    "    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)"
)
if old_init in content:
    content = content.replace(old_init, new_no_init, 1)
    print('OK: Removed dead init line')
else:
    print('WARN: dead init pattern not found')

# Step 2: Find and move the entire pretraining block BEFORE the callback construction
# The callback construction is: 
#     grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)
# And the pretraining block starts with:
#     # LSTM pretraining: give the feature extractor useful initial representations
#     try:
#         from training.pretrain_lstm import pretrain_feature_extractor
#         ...
#     except Exception as pre_err:
#         ...
# And is followed by:
#     logger.info("Starting PPO training")

# We need to find the exact boundaries and move the block up
# The pretrain block is between:
#   "if eval_callback is not None:\n        callbacks.insert(0, eval_callback)"
# and 
#   "logger.info(\"Starting PPO training\")"

# Let me find the pretrain block start
pretrain_start_marker = "# LSTM pretraining: give the feature extractor useful initial representations"
pretrain_end_marker = "    logger.info(\"Starting PPO training\")"

# Find positions
ps_pos = content.find(pretrain_start_marker)
pe_pos = content.find(pretrain_end_marker)
if ps_pos >= 0 and pe_pos > ps_pos:
    # Extract the pretrain block (everything from the marker to just before logger.info)
    pretrain_block = content[ps_pos:pe_pos]
    
    # Find the callback construction line
    cb_line = "    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)"
    cb_pos = content.find(cb_line)
    
    if cb_pos >= 0 and cb_pos < ps_pos:
        # Remove pretrain block from its current position
        content_without_pretrain = content[:ps_pos] + content[pe_pos:]
        
        # Find where to insert it (before the callback line, respecting the new position)
        # After removal, the callback line position may have shifted
        cb_pos_after = content_without_pretrain.find(cb_line)
        
        if cb_pos_after >= 0:
            # Insert pretrain block before the callback line
            content_final = (
                content_without_pretrain[:cb_pos_after] 
                + pretrain_block 
                + content_without_pretrain[cb_pos_after:]
            )
            
            # Verify the ordering: pretrain should come before callback
            ps_new = content_final.find(pretrain_start_marker)
            cb_new = content_final.find(cb_line)
            if ps_new >= 0 and cb_new >= 0 and ps_new < cb_new:
                print('OK: Moved pretrain block BEFORE callback construction')
                print(f'  pretrain at pos {ps_new}, callback at pos {cb_new}')
                content = content_final
            else:
                print(f'WARN: ordering not preserved! pretrain={ps_new}, callback={cb_new}')
                # Fall back to simple approach
                print('  Falling back to direct attribute approach')
                content = None
        else:
            print('WARN: callback line not found after removal')
            content = None
    else:
        print(f'WARN: callback line not found before pretrain block (cb={cb_pos}, ps={ps_pos})')
        content = None
else:
    print(f'WARN: pretrain block boundaries not found (ps={ps_pos}, pe={pe_pos})')
    content = None

if content is not None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print('OK: train_drl.py updated')
else:
    print('FAIL: could not restructure - trying attribute fallback approach')
