"""YOLO26-N with Neck Width 0.5 + GhostConv (Track 2, H6).

Neck channels halved relative to baseline. Full FPN+PAN preserved.
Detect channels: [32, 64, 128] (vs baseline [64, 128, 256]).
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA
from ultralytics.nn.modules.head import Detect

from .modules.ghost_c3k2 import GhostC3k2


class YOLO26NeckWide(nn.Module):
    """YOLO26-Nano with neck channels halved + GhostConv."""

    def __init__(self, nc=80, reg_max=1, end2end=True):
        super().__init__()
        self.nc = nc
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        w, d, max_c = 0.25, 0.50, 1024

        def _w(c): return min(int(c * w), max_c)
        def _d(n): return max(round(n * d), 1)

        # Backbone — standard, unchanged
        self.b0 = Conv(3, _w(64), 3, 2)
        self.b1 = Conv(_w(64), _w(128), 3, 2)
        self.b2 = C3k2(_w(128), _w(256), _d(2), False, 0.25)
        self.b3 = Conv(_w(256), _w(256), 3, 2)
        self.b4 = C3k2(_w(256), _w(512), _d(2), False, 0.25)   # P3, out=128
        self.b5 = Conv(_w(512), _w(512), 3, 2)
        self.b6 = C3k2(_w(512), _w(512), _d(2), True)            # P4, out=128
        self.b7 = Conv(_w(512), _w(1024), 3, 2)
        self.b8 = C3k2(_w(1024), _w(1024), _d(2), True)
        self.b9 = SPPF(_w(1024), _w(1024), 5, 3, True)
        self.b10 = C2PSA(_w(1024), _w(1024), _d(2))              # P5, out=256

        # Neck — 0.5× channels + GhostConv
        # Channel budget: we halve the *output* channels of each neck C3k2
        # Input = concat of neck-up + backbone (inputs stay the same size, we halve the output)
        neck_scale = 0.5
        def _nw(c): return min(int(c * w * neck_scale), max_c)

        # FPN input channels remain same (concat of backbone outputs)
        # FPN output channels are halved
        n11_out = _nw(512)   # 64
        n12_out = _nw(256)   # 32

        # PAN: input = Concat(n12_out + n11_out), or Concat(down + up)
        # n13 input = n12_out (from down) + n11_out (from FPN P4) = 32+64=96
        n13_in = n12_out + n11_out  # 96
        n13_out = _nw(512)          # 64

        # n14 input = n13_out (from down) + _w(1024)(from b10) = 64+256=320
        n14_in = n13_out + _w(1024)  # 320
        n14_out = _nw(1024)          # 128

        self.n_up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.n_cat1 = _Concat()
        self.n11 = GhostC3k2(_w(1024) + _w(512), n11_out, _d(2), True)

        self.n_up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.n_cat2 = _Concat()
        self.n12 = GhostC3k2(n11_out + _w(512), n12_out, _d(2), True)

        self.n_down1 = Conv(n12_out, n12_out, 3, 2)
        self.n_cat3 = _Concat()
        self.n13 = GhostC3k2(n13_in, n13_out, _d(2), True)

        self.n_down2 = Conv(n13_out, n13_out, 3, 2)
        self.n_cat4 = _Concat()
        self.n14 = GhostC3k2(n14_in, n14_out, _d(1), True)

        ch = [n12_out, n13_out, n14_out]  # [32, 64, 128]
        self.detect = Detect(nc, reg_max=reg_max, end2end=end2end, ch=ch)

    def forward(self, x):
        b0 = self.b0(x); b1 = self.b1(b0); b2 = self.b2(b1)
        b3 = self.b3(b2); b4 = self.b4(b3); b5 = self.b5(b4)
        b6 = self.b6(b5); b7 = self.b7(b6); b8 = self.b8(b7)
        b9 = self.b9(b8); b10 = self.b10(b9)

        n = self.n_up1(b10); n = self.n_cat1([n, b6]); n11 = self.n11(n)
        n = self.n_up2(n11); n = self.n_cat2([n, b4]); p3 = self.n12(n)
        n = self.n_down1(p3); n = self.n_cat3([n, n11]); p4 = self.n13(n)
        n = self.n_down2(p4); n = self.n_cat4([n, b10]); p5 = self.n14(n)

        return self.detect([p3, p4, p5])


class _Concat(nn.Module):
    def forward(self, x): return torch.cat(x, 1)