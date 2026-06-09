import sys
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add pretrain_loss_reduction = 0.0 before the try block
old = (
    '# LSTM pretraining: give the feature extractor useful initial representations\n'
    '    try:\n'
    '        from training.pretrain_lstm import pretrain_feature_extractor'
)
new = (
    '# LSTM pretraining: give the feature extractor useful initial representations\n'
    '    pretrain_loss_reduction = 0.0\n'
    '    try:\n'
    '        from training.pretrain_lstm import pretrain_feature_extractor'
)
if old in content:
    content = content.replace(old, new, 1)
    print('OK: Added pretrain_loss_reduction = 0.0 before try block')
else:
    print('WARN: Pattern not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: train_drl.py updated')
