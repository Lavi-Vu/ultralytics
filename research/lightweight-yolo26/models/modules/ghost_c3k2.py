"""C3k2 variant with GhostConv bottlenecks.

Replaces the standard Bottleneck inside C3k2 with GhostBottleneck
for ~2× FLOP reduction in the neck while preserving feature quality.
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv

from .ghost_bottleneck import GhostBottleneck


class GhostC3k2(nn.Module):
    """C3k2 variant where all Bottlenecks use GhostConv internally.

    Matches C3k2 signature:
      GhostC3k2(c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True)
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            GhostBottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))