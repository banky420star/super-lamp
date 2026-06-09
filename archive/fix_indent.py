import sys
path = r'C:\supreme-chainsaw\training\train_drl.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The bug: the try/except NUM_REGIMES block is outside the function (no indent)
# Need to indent it properly inside _policy_kwargs_for

old = (
    "    from drl.lstm_feature_extractor import LSTMFeatureExtractor\ntry:\n    from drl.regime_detector import NUM_REGIMES\nexcept Exception:\n    NUM_REGIMES = 5"
)
new = (
    "    from drl.lstm_feature_extractor import LSTMFeatureExtractor\n    try:\n        from drl.regime_detector import NUM_REGIMES\n    except Exception:\n        NUM_REGIMES = 5"
)

if old in content:
    content = content.replace(old, new, 1)
    print('OK: Fixed indent of NUM_REGIMES import')
else:
    print('WARN: Pattern not found - trying different approach')
    # Find and fix the unindented try block
    import re
    # Look for "try:" at the start of a line (no indent) inside the function
    idx = content.find('try:\n    from drl.regime_detector')
    if idx >= 0:
        # Find the preceding newline
        prev_newline = content.rfind('\n', 0, idx)
        prev_content = content[prev_newline+1:prev_newline+20]
        print(f'  Found try block at {idx}, preceding content: {repr(prev_content)}')
        print('  Manually replacing...')
        # Replace with properly indented version
        old_block = 'try:\n    from drl.regime_detector import NUM_REGIMES\nexcept Exception:\n    NUM_REGIMES = 5'
        new_block = '    try:\n        from drl.regime_detector import NUM_REGIMES\n    except Exception:\n        NUM_REGIMES = 5'
        if old_block in content:
            content = content.replace(old_block, new_block, 1)
            print('  Fixed via block replace')
        else:
            print(f'  Could not find exact block')

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)
print('OK: train_drl.py updated')
