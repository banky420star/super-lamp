import sys

# Fix 1: pretrain_lstm.py default parameter
path1 = r'C:\supreme-chainsaw\training\pretrain_lstm.py'
with open(path1, 'r', encoding='utf-8') as f:
    c1 = f.read()
c1 = c1.replace('lr: float = 1e-3', 'lr: float = 1e-4')
with open(path1, 'w', encoding='utf-8') as f:
    f.write(c1)
print('OK: pretrain_lstm.py default LR 1e-3 -> 1e-4')

# Fix 2: train_drl.py call site
path2 = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path2, 'r', encoding='utf-8') as f:
    c2 = f.read()
c2 = c2.replace('lr=1e-3', 'lr=1e-4')
with open(path2, 'w', encoding='utf-8') as f:
    f.write(c2)
print('OK: train_drl.py call site LR 1e-3 -> 1e-4')

print('Done')
