"""Adaptive channel gating module.

Learns a per-channel gating vector g ∈ R^C.
During training: soft gating via sigmoid(g) with L1 sparsity loss.
During inference: hard gate — channels with sigmoid(g) > 0.5 are kept, others zeroed.

This is NOT the same as SE/CBAM attention — it produces a hard-or-soft mask,
not a reweighting. The gate is regularized toward binary decisions via L1.
"""

import torch
import torch.nn as nn


class ChannelGate(nn.Module):
    """Learnable per-channel gate that adaptively prunes channels.

    Args:
        channels: Number of input/output channels
        init_keep_prob: Initial probability of keeping a channel (0.0–1.0)
        regularize: Whether to apply L1 sparsity loss on gate logits
    """

    def __init__(self, channels, init_keep_prob=0.85, regularize=True):
        super().__init__()
        self.channels = channels
        self.regularize = regularize
        # Learnable logit per channel
        logit_init = -torch.log(torch.tensor(1.0 / init_keep_prob - 1.0))
        self.gate_logits = nn.Parameter(torch.full((1, channels, 1, 1), logit_init))

    def forward(self, x):
        if self.training:
            # Soft gate during training
            gate = torch.sigmoid(self.gate_logits)
            return x * gate
        else:
            # Hard gate during inference
            gate = (torch.sigmoid(self.gate_logits) > 0.5).float()
            return x * gate

    def sparsity_loss(self):
        """L1 sparsity regularization loss — encourages binary gate decisions."""
        if not self.regularize or not self.training:
            return 0.0
        gate = torch.sigmoid(self.gate_logits)
        return gate.mean()  # lower mean = more channels zeroed

    def frac_active(self):
        """Fraction of channels kept (sigmoid(g) > 0.5) — for logging."""
        with torch.no_grad():
            gate = torch.sigmoid(self.gate_logits)
            return (gate > 0.5).float().mean().item()

    def extra_repr(self):
        return f"channels={self.channels}, regularize={self.regularize}"