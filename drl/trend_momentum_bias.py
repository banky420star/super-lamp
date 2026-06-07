"""
Trend + Momentum Bias Layer

Computes a directional market-bias signal from LSTM-extracted features,
providing the policy with a soft directional prior. Instead of treating
all features equally, the policy can condition its decisions on whether
the market is trending up/down, ranging, or unstable.

Architecture placement:
    AdaptiveLSTMFeatureExtractor
        -> TrendMomentumBiasLayer (NEW)
        -> FeatureGroupGate
        -> RegimeRoutedPolicy
"""

import torch as th
import torch.nn as nn


class TrendMomentumBiasLayer(nn.Module):
    """
    Computes directional market bias from LSTM-extracted features.

    Adds a soft directional prior so the policy can distinguish between
    bullish, bearish, ranging, and unstable market states rather than
    treating all feature combinations equally.

    Produces 6 bias features appended to the input:
        col 0: trend_score        [-1 .. +1]  direction of trend
        col 1: momentum_score     [-1 .. +1]  force behind movement
        col 2: direction_bias     [-1 .. +1]  combined (trend+momentum)/2
        col 3: confidence         [ 0 ..  1]  how reliable the bias is
        col 4: agreement          [ 0 ..  1]  trend-momentum alignment
        col 5: persistent_bias    [-1 .. +1]  hysteresis-smoothed bias
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.num_bias_features = 6

        # Trend score: small MLP -> Tanh (-1 to +1)
        self.trend_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # Momentum score: small MLP -> Tanh (-1 to +1)
        self.momentum_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # Confidence: small MLP -> Sigmoid (0 to 1)
        self.confidence_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Persistent bias state for hysteresis smoothing during eval
        self.register_buffer('_persistent_bias', th.zeros(1))

    def forward(self, features: th.Tensor, use_persistence: bool = True) -> th.Tensor:
        """
        Args:
            features: (batch, input_dim) -- projected features from LSTM extractor.
            use_persistence: apply hysteresis smoothing during single-step eval.

        Returns:
            (batch, input_dim + 6) -- original features with 6 bias signals appended.
        """
        trend = self.trend_net(features)            # (batch, 1)
        momentum = self.momentum_net(features)       # (batch, 1)
        direction_bias = (trend + momentum) / 2.0    # (batch, 1)
        confidence = self.confidence_net(features)   # (batch, 1)
        agreement = 1.0 - th.abs(trend - momentum) / 2.0  # (batch, 1)

        batch_size = features.shape[0]
        if use_persistence and not self.training:
            if batch_size == 1:
                # Single-step eval: hysteresis to prevent flip-flopping
                diff = direction_bias - self._persistent_bias
                d = diff[0, 0].item()
                c = confidence[0, 0].item()
                if c > 0.35 and abs(d) > 0.15:
                    # Smooth transition: blend 40% toward new bias
                    persistent_bias = self._persistent_bias + diff * 0.4
                else:
                    # Stay with current bias
                    persistent_bias = self._persistent_bias.clone()
                self._persistent_bias = persistent_bias.detach()
            else:
                # Batched eval: no per-sample state, use raw bias
                persistent_bias = direction_bias
        else:
            persistent_bias = direction_bias

        bias_features = th.cat(
            [trend, momentum, direction_bias, confidence, agreement, persistent_bias],
            dim=1,
        )
        return th.cat([features, bias_features], dim=1)

    def reset_persistent_bias(self):
        """Reset hysteresis state -- call on environment reset during live trading."""
        self._persistent_bias.fill_(0.0)
