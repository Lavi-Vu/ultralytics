"""Reparameterized convolution block for backbone Bottleneck replacement.

Train-time: multi-branch (3x3 conv + 1x1 conv + optional identity).
Infer-time: fused single 3x3 conv with zero FLOP overhead.
Based on RepVGG (Ding et al., 2021) and DBB (Li et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ReparamConvBlock(nn.Module):
    """Single reparameterizable convolution block — trains as multi-branch, infers as single 3x3 Conv."""

    def __init__(self, c1, c2, k=3, s=1, g=1):
        super().__init__()
        assert k == 3 and s == 1, "ReparamConvBlock only supports k=3, s=1 (for Bottleneck use)"
        self.c1 = c1
        self.c2 = c2
        self.g = g

        # Branch 1: 3x3 Conv
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c1, c2, 3, s, 1, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        )
        # Branch 2: 1x1 Conv (padded to 3x3 at fusion)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, s, 0, groups=g, bias=False),
            nn.BatchNorm2d(c2),
        )
        # Branch 3: Identity (only if c1 == c2 and s == 1)
        self.id = nn.BatchNorm2d(c2) if c1 == c2 and s == 1 else None

    def forward(self, x):
        out = self.conv_3x3(x) + self.conv_1x1(x)
        if self.id is not None:
            out = out + self.id(x)
        return out

    @torch.no_grad()
    def fuse_convs(self):
        """Fuse all branches into a single 3x3 Conv for inference."""
        if not hasattr(self, "conv_3x3"):
            return  # already fused
        kernel_3x3, bias_3x3 = self._fuse_bn(self.conv_3x3[0], self.conv_3x3[1])
        kernel_1x1, bias_1x1 = self._fuse_bn(self.conv_1x1[0], self.conv_1x1[1])
        kernel_1x1 = self._pad_1x1_to_3x3(kernel_1x1)

        kernel_id, bias_id = 0, 0
        if self.id is not None:
            kernel_id, bias_id = self._fuse_bn_identity(self.id)

        fused_kernel = kernel_3x3 + kernel_1x1 + kernel_id
        fused_bias = bias_3x3 + bias_1x1 + bias_id

        conv = nn.Conv2d(
            self.c1, self.c2, 3, 1, 1, groups=self.g, bias=True
        ).to(fused_kernel.device)
        conv.weight.data.copy_(fused_kernel)
        conv.bias.data.copy_(fused_bias)

        # Replace multi-branch with single conv
        self.conv = conv
        del self.conv_3x3
        del self.conv_1x1
        if hasattr(self, "id"):
            del self.id
        self.forward = self._forward_fused

    def _forward_fused(self, x):
        return self.conv(x)

    @staticmethod
    def _fuse_bn(conv, bn):
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _fuse_bn_identity(self, bn):
        if bn is None:
            return 0, 0
        input_dim = self.c2 // self.g
        kernel_val = np.zeros((self.c2, input_dim, 3, 3), dtype=np.float32)
        for i in range(self.c2):
            kernel_val[i, i % input_dim, 1, 1] = 1
        kernel = torch.from_numpy(kernel_val).to(bn.weight.device)
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    @staticmethod
    def _pad_1x1_to_3x3(kernel):
        if kernel is None:
            return 0
        return F.pad(kernel, [1, 1, 1, 1])


class ReparamBottleneck(nn.Module):
    """Bottleneck block with reparameterizable convolutions.

    Matches the standard Ultralytics Bottleneck signature:
      Conv(c, c, 3) -> Conv(c, c, 3)  with optional shortcut.
    Both convs are reparameterized (train multi-branch, infer single-branch).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        self.cv1 = ReparamConvBlock(c1, c2, k=3, s=1, g=g)
        self.cv2 = ReparamConvBlock(c2, c2, k=3, s=1, g=g)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))

    @torch.no_grad()
    def fuse_convs(self):
        self.cv1.fuse_convs()
        self.cv2.fuse_convs()