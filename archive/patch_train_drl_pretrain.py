import sys
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r') as f:
    content = f.read()

# Replace the pretrain call to capture the result
old_call = (
    '            pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
    '            # Reset env after pretraining'
)
new_call = (
    '            pretrain_result = pretrain_feature_extractor(model, env, steps=pretrain_steps, lr=1e-3, mode="ae", logger_obj=logger)\n'
    '            pretrain_loss_reduction = pretrain_result[1] if isinstance(pretrain_result, tuple) and pretrain_result[0] else 0.0\n'
    '            # Reset env after pretraining'
)
if old_call in content:
    content = content.replace(old_call, new_call, 1)
    print('OK: pretrain call updated to capture return value')
else:
    print('WARN: pretrain call pattern not found!')
    import re
    matches = list(re.finditer(r'pretrain_feature_extractor\(', content))
    for m in matches:
        start = max(0, m.start() - 50)
        end = min(len(content), m.end() + 100)
        print(f'  Found at pos {m.start()}: {repr(content[start:end])}')

# Replace the LSTMGradientDiagnostics construction to pass pretrain_loss_reduction
old_grad = '    grad_callback = LSTMGradientDiagnostics()'
new_grad = '    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)'
if old_grad in content:
    content = content.replace(old_grad, new_grad, 1)
    print('OK: LSTMGradientDiagnostics construction updated')
else:
    print('WARN: LSTMGradientDiagnostics pattern not found!')

with open(path, 'w') as f:
    f.write(content)
print('OK: train_drl.py updated')
