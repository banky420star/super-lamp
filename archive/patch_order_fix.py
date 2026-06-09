import sys
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add initialization of pretrain_loss_reduction BEFORE grad_callback line
old = (
    "    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)"
)
new = (
    "    pretrain_loss_reduction = 0.0  # will be updated by LSTM pretrain below if enabled\n"
    "    grad_callback = LSTMGradientDiagnostics(pretrain_loss_reduction=pretrain_loss_reduction)"
)
if old in content:
    content = content.replace(old, new, 1)
    print('OK: Added pretrain_loss_reduction initialization before LSTMGradientDiagnostics')
else:
    print('WARN: Pattern not found!')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: train_drl.py fix applied')
