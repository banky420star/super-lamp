#!/usr/bin/env python3
"""Patch AdaptiveLSTMFeatureExtractor to inject regime features into LSTM."""
import re

path = r"C:\supreme-chainsaw\drl\adaptive_feature_extractor.py"

with open(path, "r") as f:
    content = f.read()

old_sig = (
    'class AdaptiveLSTMFeatureExtractor(BaseFeaturesExtractor):\n'
    '    """\n'
    '    Trainable extractor for newer feature contracts where the live AGI LSTM\n'
    '    bundle should not dictate PPO observation handling.\n'
    '\n'
    '    v2: Bidirectional LSTM - processes the time-window forward AND backward\n'
    '    so the representation at each step benefits from both past AND future\n'
    '    context. Output dimension is doubled (hidden_size * 2) before projection.\n'
    '    """\n'
    '\n'
    '    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, window_size: int = 100, num_heads: int = 4):'
)

new_sig = (
    'class AdaptiveLSTMFeatureExtractor(BaseFeaturesExtractor):\n'
    '    """\n'
    '    Trainable extractor for newer feature contracts where the live AGI LSTM\n'
    '    bundle should not dictate PPO observation handling.\n'
    '\n'
    '    v2: Bidirectional LSTM - processes the time-window forward AND backward\n'
    '    so the representation at each step benefits from both past AND future\n'
    '    context. Output dimension is doubled (hidden_size * 2) before projection.\n'
    '\n'
    '    v5 (Regime Injection Fix): When regime_dim > 0, regime features\n'
    '    (one-hot + confidence, at the end of the observation tail) are extracted,\n'
    '    repeated across the sequence dimension, and concatenated to the per-bar\n'
    '    features BEFORE the LSTM. This lets the LSTM learn regime-conditional\n'
    '    temporal patterns. Portfolio state (non-regime tail) is still\n'
    '    concatenated after the LSTM as before.\n'
    '    """\n'
    '\n'
    '    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, window_size: int = 100, num_heads: int = 4, regime_dim: int = 0):'
)

if old_sig not in content:
    print("ERR: could not find exact old signature")
    # Show a snippet for debugging
    idx = content.find('class AdaptiveLSTMFeatureExtractor')
    if idx >= 0:
        print("Found at", idx)
        print(repr(content[idx:idx+300]))
    raise SystemExit(1)

content = content.replace(old_sig, new_sig, 1)

# Now replace the __init__ body - find the exact ranges
lines = content.split("\n")

# Find __init__ body
init_start = None
init_end = None
for i, line in enumerate(lines):
    if line.strip().startswith('def __init__'):
        init_start = i
    if init_start is not None and i > init_start and line.strip().startswith('def '):
        init_end = i
        break

# New __init__ body
new_init_body = [
    '        total_obs = int(observation_space.shape[0])',
    '        self.seq_window = int(window_size)',
    '        self.regime_dim = int(regime_dim)',
    '        self.portfolio_dim = total_obs % self.seq_window',
    '        seq_flat = total_obs - self.portfolio_dim',
    '        if seq_flat <= 0 or seq_flat % self.seq_window != 0:',
    '            raise ValueError(',
    '                f"Invalid observation shape for AdaptiveLSTMFeatureExtractor: total={total_obs}, "',
    '                f"window={self.seq_window}, portfolio_dim={self.portfolio_dim}"',
    '            )',
    '',
    '        # Portfolio dim excluding regime (regime goes into LSTM instead)',
    '        actual_portfolio_dim = max(0, self.portfolio_dim - self.regime_dim)',
    '',
    '        super().__init__(observation_space, features_dim=features_dim + actual_portfolio_dim)',
    '        self.seq_feature_dim = seq_flat // self.seq_window',
    '',
    '        # LSTM input: base features + regime features per bar',
    '        lstm_input_dim = self.seq_feature_dim + self.regime_dim',
    '        self.encoder = torch.nn.LSTM(',
    '            input_size=lstm_input_dim,',
    '            hidden_size=160,',
    '            num_layers=2,',
    '            dropout=0.2,',
    '            batch_first=True,',
    '            bidirectional=True,',
    '        )',
    '        self.lstm_hidden = 320',
    '        self.projection = torch.nn.Sequential(',
    '            torch.nn.Linear(self.lstm_hidden, features_dim),',
    '            torch.nn.LeakyReLU(negative_slope=0.01),',
    '            torch.nn.Linear(features_dim, features_dim),',
    '        )',
    '        self.attention = MultiHeadAttentionPooling(self.lstm_hidden, num_heads=num_heads)',
    '        self.lstm_norm = torch.nn.LayerNorm(self.lstm_hidden)',
]

# Find forward method body
fwd_start = None
fwd_end = None
for i, line in enumerate(lines):
    if line.strip().startswith('def forward'):
        fwd_start = i
    if fwd_start is not None and i > fwd_start and line.strip().startswith('def '):
        fwd_end = i
        break

new_fwd_body = [
    '        batch_size = observations.shape[0]',
    '        seq_features = observations[:, :-self.portfolio_dim] if self.portfolio_dim else observations',
    '        tail = observations[:, -self.portfolio_dim:] if self.portfolio_dim else observations.new_zeros((batch_size, 0))',
    '',
    '        # Split tail: regime at END (last regime_dim), portfolio is the rest',
    '        if self.regime_dim > 0 and tail.shape[-1] >= self.regime_dim:',
    '            regime = tail[:, -self.regime_dim:]',
    '            portfolio_state = tail[:, :-self.regime_dim]',
    '        else:',
    '            regime = None',
    '            portfolio_state = tail',
    '',
    '        seq = seq_features.view(batch_size, self.seq_window, self.seq_feature_dim)',
    '',
    '        # Inject regime into each timestep BEFORE the LSTM',
    '        if regime is not None:',
    '            regime_expanded = regime.unsqueeze(1).expand(-1, self.seq_window, -1)',
    '            seq = torch.cat([seq, regime_expanded], dim=-1)',
    '',
    '        encoded, _ = self.encoder(seq)',
    '        lstm_out = self.attention(encoded)',
    '        lstm_out = self.lstm_norm(lstm_out)',
    '        projected = self.projection(lstm_out)',
    '',
    '        # Only concat portfolio (regime is already in LSTM representation)',
    '        if portfolio_state.shape[-1] > 0:',
    '            return torch.cat([projected, portfolio_state], dim=1)',
    '        return projected',
]

# Rebuild the file
new_lines = []
i = 0
while i < len(lines):
    if i == init_start:
        new_lines.append(lines[i])  # def __init__ line
        for line in new_init_body:
            new_lines.append(line)
        i = init_end  # skip old init body
    elif i == fwd_start:
        new_lines.append(lines[i])  # def forward line
        for line in new_fwd_body:
            new_lines.append(line)
        i = fwd_end  # skip old forward body
    else:
        new_lines.append(lines[i])
        i += 1

new_content = "\n".join(new_lines)

try:
    compile(new_content, path, "exec")
    with open(path, "w") as f:
        f.write(new_content)
    print(f"OK: Applied regime injection fix to {path}")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    nl = new_content.split("\n")
    if e.lineno:
        for j in range(max(0, e.lineno - 5), min(len(nl), e.lineno + 3)):
            marker = ">>>" if j == e.lineno - 1 else "   "
            print(f"  {marker} {j+1}: {nl[j]}")
