import sys
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Add safe NUM_REGIMES import near the other imports
# Find the import section and add NUM_REGIMES after the last regime_detector import or similar
old = (
    'from drl.lstm_feature_extractor import LSTMFeatureExtractor'
)
new = (
    'from drl.lstm_feature_extractor import LSTMFeatureExtractor\n'
    'try:\n'
    '    from drl.regime_detector import NUM_REGIMES\n'
    'except Exception:\n'
    '    NUM_REGIMES = 5'
)
if old in content:
    content = content.replace(old, new, 1)
    print('OK: Added NUM_REGIMES import')
else:
    # Try to find nearby import
    import re
    matches = list(re.finditer(r'from drl\.\w+ import', content))
    for m in matches:
        print(f'  Found: {content[m.start():m.end()]} at {m.start()}')
    print('WARN: Could not find target import')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: train_drl.py updated')
