import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback
import os

os.makedirs("/tmp/logs/gradients", exist_ok=True)
writer = SummaryWriter("/tmp/logs/gradients")

class LSTMGradientDiagnostics(BaseCallback):
    """Real-time LSTM gradient flow diagnostics (works perfectly with joint training)"""
    def __init__(self, verbose=0, pretrain_loss_reduction=None):
        super().__init__(verbose)
        self.epoch = 0
        self.pretrain_loss_reduction = pretrain_loss_reduction

    def _on_step(self) -> bool:
        # Run diagnostics every 5 rollouts
        if self.n_calls % (self.model.n_steps * 5) == 0:
            self.epoch += 1
            self.analyze(self.model.policy, self.epoch)
        return True

    def analyze(self, policy, epoch: int):
        lstm_norms = []
        ppo_norms = []

        for name, param in policy.named_parameters():
            if param.grad is None or param.grad.data.numel() == 0:
                continue
            norm = param.grad.data.norm(2).item()

            if any(x in name.lower() for x in ["lstm", "lstm_brain"]):
                lstm_norms.append(norm)
                writer.add_scalar(f"grad_flow/lstm/{name.split('.')[-1]}", norm, epoch)
            else:
                ppo_norms.append(norm)
                writer.add_scalar(f"grad_flow/ppo/{name.split('.')[-1]}", norm, epoch)

        if lstm_norms:
            mean_lstm = np.mean(lstm_norms)
            max_lstm = np.max(lstm_norms)
            min_lstm = np.min(lstm_norms)

            logger.info(f"📊 EPOCH {epoch:04d} | LSTM GRADIENT DIAGNOSTICS")
            logger.info(f"   Mean: {mean_lstm:.2e} | Max: {max_lstm:.2e} | Min: {min_lstm:.2e}")

            if mean_lstm < 1e-8:
                logger.error("🚨 VANISHING GRADIENTS IN LSTM — reduce LSTM LR immediately")
            elif mean_lstm > 5.0:
                logger.error("🚨 EXPLODING GRADIENTS — add gradient clipping")
            elif 1e-5 < mean_lstm < 0.8:
                logger.success("✅ EXCELLENT LSTM gradient flow — brain is learning trading memory!")
            else:
                logger.warning("⚠️ Moderate LSTM flow — still healthy but monitor")

        writer.flush()



class DiagnosticsCallback(BaseCallback):
    """Logs basic training diagnostics at fixed intervals."""
    def __init__(self, log_interval: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_interval = log_interval
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log >= self.log_interval:
            self._last_log = self.num_timesteps
            if self.verbose > 0:
                print(f"[Diagnostics] step={self.num_timesteps}, reward={self.locals.get('rewards', [None])[0] if 'rewards' in self.locals else 'N/A'}")
        return True
