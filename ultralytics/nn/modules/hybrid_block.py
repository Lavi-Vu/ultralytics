# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Hybrid CNN-Transformer blocks for EdgeRF-YOLO architecture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .conv import Conv
from .light_transformer import (
    InterleavedTransformerStage,
    LightTransformerBlock,
    ConvLayerNorm,
)

__all__ = (
    "C2fHybrid",
    "C2fHybridGlobal",
    "EdgeRFStage",
    "LightTransformerStage",
)


class C2fHybrid(nn.Module):
    """C2f-like block with hybrid CNN + windowed transformer layers.
    
    Extends the C2f paradigm by replacing standard Bottleneck blocks
    with lightweight windowed transformer blocks for enhanced local
    feature interaction at lower resolution stages.
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of hybrid blocks.
        shortcut (bool): Use residual connections.
        g (int): Groups for convolutions.
        e (float): Expansion ratio.
        num_heads (int): Attention heads.
        window_size (int): Window size for local attention.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, num_heads=4, window_size=7, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            LightTransformerBlock(
                dim=self.c,
                num_heads=min(num_heads, max(self.c // 32, 1)),
                window_size=window_size,
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_fuse(self, x):
        return self.forward(x)


class C2fHybridGlobal(nn.Module):
    """C2f-like block with hybrid CNN + global transformer layers.
    
    Uses spatially-reduced global attention for higher-level stages,
    enabling long-range dependency modeling.
    
    Args:
        c1 (int): Input channels.
        c2 (int): Output channels.
        n (int): Number of hybrid blocks.
        shortcut (bool): Use residual connections.
        g (int): Groups for convolutions.
        e (float): Expansion ratio.
        num_heads (int): Attention heads.
        sr_ratio (int): Spatial reduction ratio for global attention.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5, num_heads=4, sr_ratio=2, g=1):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            LightTransformerBlock(
                dim=self.c,
                num_heads=min(num_heads, max(self.c // 32, 1)),
                sr_ratio=sr_ratio,
            )
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_fuse(self, x):
        return self.forward(x)


class LightTransformerStage(nn.Module):
    """Standalone interleaved transformer stage for deeper backbone layers.
    
    Used as a drop-in replacement for CNN stages in the backbone.
    Includes a channel projection for input/output adaptation.
    """

    def __init__(self, c1, c2, n=2, num_heads=4, window_size=7,
                 sr_ratio=2, mlp_ratio=2.0):
        super().__init__()
        self.proj_in = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.stage = InterleavedTransformerStage(
            dim=c2,
            num_heads=num_heads,
            window_size=window_size,
            sr_ratio=sr_ratio,
            num_layers=n,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        x = self.proj_in(x)
        return self.stage(x)

    def forward_fuse(self, x):
        return self.forward(x)


class EdgeRFStage(nn.Module):
    """EdgeRF stage combining CNN downsampling + interleaved transformers.
    
    Designed as a backbone stage for EdgeRF-YOLO, including:
    1. Optional strided conv for spatial downsampling
    2. Interleaved window/global transformer blocks
    """

    def __init__(self, c1, c2, n=2, stride=2, num_heads=4,
                 window_size=7, sr_ratio=2, mlp_ratio=2.0):
        super().__init__()
        self.downsample = Conv(c1, c2, 3, stride) if stride > 1 else Conv(c1, c2, 1)
        self.stage = InterleavedTransformerStage(
            dim=c2,
            num_heads=num_heads,
            window_size=window_size,
            sr_ratio=sr_ratio,
            num_layers=n,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x):
        x = self.downsample(x)
        return self.stage(x)

    def forward_fuse(self, x):
        return self.forward(x)
