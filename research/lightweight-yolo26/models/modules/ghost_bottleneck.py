"""Ghost bottleneck for use inside C3k2 blocks.

GhostConv generates c2//2 channels via standard Conv and c2//2 via cheap 5x5 depthwise.
This 2× speedup over standard Conv with minimal accuracy loss.

Reference: GhostNet (Han et al., 2020 CVPR).
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv, DWConv


class GhostBottleneck(nn.Module):
    """Bottleneck replacement using GhostConv internally.

    Matches standard Bottleneck interface:
      Conv(c, c, 3) -> Conv(c, c, 3)  with optional shortcut.
    But each Conv is replaced by GhostConv (standard + cheap depthwise).
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=1.0):
        super().__init__()
        self.cv1 = GhostConvReparam(c1, c2, k=3, s=1)
        self.cv2 = GhostConvReparam(c2, c2, k=3, s=1)
        self.shortcut = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class GhostConvReparam(nn.Module):
    """GhostConv with reparameterizable 5x5 vs 3x3 primary path.

    Primary:  Conv(c1, c_, 3)    → c_ = c2 // 2 channels
    Cheap:    DWConv(c_, c_, 5)  → depthwise 5x5 on c_ channels
    Concatenate along channel dim → c2 channels.

    During training: both branches active.
    During inference: can fuse primary BN into conv.
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, act=False)  # primary
        self.cv2 = DWConv(c_, c_, 5, 1, act=False)  # cheap
        self.act = nn.SiLU()

    def forward(self, x):
        y = self.cv1(x)
        return self.act(torch.cat([y, self.cv2(y)], 1))

    def forward_fuse(self, x):
        """Fused forward (after fusing BN into conv in cv1)."""
        y = self.cv1.forward_fuse(x)
        return self.act(torch.cat([y, self.cv2(y)], 1))