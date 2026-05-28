"""YOLO26-N Baseline — exact channel-correct reproduction.

Neck channel trace (width=0.25):
  b10(P5)=256, b6(P4)=128, b4(P3)=128
  n11: concat(256+128)=384 → c2=_w(512)=128
  n12: concat(128+128)=256 → c2=_w(256)=64
  n13: concat(64+128)=192 → c2=_w(512)=128
  n14: concat(128+256)=384 → c2=_w(1024)=256
  Detect ch = [64, 128, 256]
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C3k2, SPPF, C2PSA
from ultralytics.nn.modules.head import Detect


class YOLO26NBaseline(nn.Module):
    """Baseline YOLO26-Nano. Matches ultralytics YOLO26n exactly."""

    def __init__(self, nc=80, reg_max=1, end2end=True):
        super().__init__()
        self.nc = nc
        self.stride = torch.tensor([8.0, 16.0, 32.0])
        w, d, max_c = 0.25, 0.50, 1024

        def _w(c): return min(int(c * w), max_c)
        def _d(n): return max(round(n * d), 1)

        # Backbone
        self.b0 = Conv(3, _w(64), 3, 2)
        self.b1 = Conv(_w(64), _w(128), 3, 2)
        self.b2 = C3k2(_w(128), _w(256), _d(2), False, 0.25)
        self.b3 = Conv(_w(256), _w(256), 3, 2)
        self.b4 = C3k2(_w(256), _w(512), _d(2), False, 0.25)      # P3, out=128
        self.b5 = Conv(_w(512), _w(512), 3, 2)
        self.b6 = C3k2(_w(512), _w(512), _d(2), True)               # P4, out=128
        self.b7 = Conv(_w(512), _w(1024), 3, 2)
        self.b8 = C3k2(_w(1024), _w(1024), _d(2), True)
        self.b9 = SPPF(_w(1024), _w(1024), 5, 3, True)
        self.b10 = C2PSA(_w(1024), _w(1024), _d(2))                 # P5, out=256

        # Neck — FPN+PAN
        self.n_up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.n_cat1 = _Concat()
        self.n11 = C3k2(_w(1024) + _w(512), _w(512), _d(2), True)   # 384→128

        self.n_up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.n_cat2 = _Concat()
        self.n12 = C3k2(_w(512) + _w(512), _w(256), _d(2), True)    # 256→64

        self.n_down1 = Conv(_w(256), _w(256), 3, 2)
        self.n_cat3 = _Concat()
        self.n13 = C3k2(_w(256) + _w(512), _w(512), _d(2), True)    # 192→128

        self.n_down2 = Conv(_w(512), _w(512), 3, 2)
        self.n_cat4 = _Concat()
        self.n14 = C3k2(_w(512) + _w(1024), _w(1024), _d(1), True)  # 384→256

        ch = [_w(256), _w(512), _w(1024)]  # [64, 128, 256]
        self.detect = Detect(nc, reg_max=reg_max, end2end=end2end, ch=ch)

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b7 = self.b7(b6)
        b8 = self.b8(b7)
        b9 = self.b9(b8)
        b10 = self.b10(b9)

        n = self.n_up1(b10); n = self.n_cat1([n, b6]); n11 = self.n11(n)
        n = self.n_up2(n11); n = self.n_cat2([n, b4]); p3 = self.n12(n)
        n = self.n_down1(p3); n = self.n_cat3([n, n11]); p4 = self.n13(n)
        n = self.n_down2(p4); n = self.n_cat4([n, b10]); p5 = self.n14(n)

        return self.detect([p3, p4, p5])


class _Concat(nn.Module):
    def forward(self, x): return torch.cat(x, 1)
    def fuse(self): return self