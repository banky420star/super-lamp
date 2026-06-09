import sys
path = r'C:\supreme-chainsaw\training\pretrain_lstm.py'
with open(path, 'r') as f:
    content = f.read()

# Update return type annotation
content = content.replace(
    ') -> bool:',
    ') -> tuple[bool, float]:',
)

# Update early returns
content = content.replace(
    '        return False\n        \n    obs_tensor',
    '        return False, 0.0\n        \n    obs_tensor',
)
content = content.replace(
    '        log.warning("No LSTM encoder, skipping pretrain")\n        return False',
    '        log.warning("No LSTM encoder, skipping pretrain")\n        return False, 0.0',
)

# Update the last return
content = content.replace(
    '    del decoder_lstm, decoder_proj\n    return True',
    '    del decoder_lstm, decoder_proj\n    return True, round(red, 1)',
)

with open(path, 'w') as f:
    f.write(content)
print('OK: pretrain_lstm.py - return changed to tuple[bool, float]')
