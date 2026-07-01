import math
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv
from .block import C2f
from .head import Detect


class ECA(nn.Module):
    """Efficient Channel Attention (ECA) module.

    Implements 1D convolution-based channel attention for efficient feature recalibration.

    Attributes:
        conv (nn.Conv1d): 1D convolution for channel attention.
        sigmoid (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        """Initialize ECA module with adaptive kernel size."""
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient channel attention to input tensor."""
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


class RepViTGhostBottleneck(nn.Module):
    """RepViTGhost bottleneck combining GhostConv, depthwise conv, and ECA attention.

    Architecture: GhostConv → DWConv3x3 → Conv1x1 → ECA → shortcut

    Attributes:
        cv1 (GhostConv | Conv): Ghost convolution for cheap feature expansion.
        cv2 (Conv): 3x3 depthwise convolution.
        cv3 (Conv): 1x1 projection convolution.
        attn (ECA): Efficient channel attention.
        add (nn.Identity | nn.Module): Shortcut connection.
    """

    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """Initialize RepViTGhostBottleneck block."""
        super().__init__()
        c_ = max(2, int(c2 * e) // 2) * 2  # ensure even for GhostConv
        if c_ >= 8:
            self.cv1 = GhostConv(c1, c_, 1, 1)
        else:
            self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=c_)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.attn = ECA(c2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GhostConv, DW, attention and optional residual connection."""
        y = self.cv3(self.cv2(self.cv1(x)))
        y = self.attn(y)
        return (x + y) if self.add else y


class C3RepViTGhost(C2f):
    """CSP bottleneck with 2 convolutions using RepViTGhostBottleneck blocks.

    This module replaces standard Bottleneck blocks with RepViTGhostBottleneck for
    lightweight feature extraction with improved efficiency.

    Attributes:
        m (nn.ModuleList): List of RepViTGhostBottleneck blocks.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, e: float = 0.5, g: int = 1):
        """Initialize C3RepViTGhost module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of RepViTGhostBottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            e (float): Expansion ratio (before g to match YAML arg order).
            g (int): Groups for convolutions.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepViTGhostBottleneck(self.c, self.c, shortcut, g) for _ in range(n))


class FastBiFusion(nn.Module):
    """Weighted bidirectional feature fusion module with learnable fusion weights.

    Fuses two feature maps using learnable scalar weights with fast normalized fusion.

    Attributes:
        cv1 (Conv | nn.Identity): 1x1 projection for first input.
        cv2 (Conv | nn.Identity): 1x1 projection for second input.
        w (nn.Parameter): Learnable fusion weights.
        conv (Conv): Post-fusion refinement convolution.
    """

    def __init__(self, c2: int, ch_in: list[int]):
        """Initialize FastBiFusion with output channels and input channel list.

        Args:
            c2 (int): Output channels after fusion.
            ch_in (list[int]): Channel sizes of the two input feature maps.
        """
        super().__init__()
        self.cv1 = Conv(ch_in[0], c2, 1, 1) if ch_in[0] != c2 else nn.Identity()
        self.cv2 = Conv(ch_in[1], c2, 1, 1) if ch_in[1] != c2 else nn.Identity()
        self.w = nn.Parameter(torch.ones(2) / 2)
        self.epsilon = 1e-4
        self.conv = Conv(c2, c2, 3, 1, g=c2) if c2 > 1 else nn.Identity()

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Fuse two feature maps with learnable weighted summation."""
        x1, x2 = x[0], x[1]
        w = F.relu(self.w)
        w_sum = w.sum() + self.epsilon
        x1 = self.cv1(x1)
        x2 = self.cv2(x2)
        y = (w[0] / w_sum) * x1 + (w[1] / w_sum) * x2
        return self.conv(y)

    def __call__(self, x: list[torch.Tensor]) -> torch.Tensor:
        """Support both list and single tensor inputs by duplicating for fusion."""
        if isinstance(x, (list, tuple)):
            return self.forward(x)
        return self.forward([x, x])


class LightEdgeDetect(Detect):
    """LightEdge detection head with improved head dimensions and joint backbone training.

    Key improvements over base Detect:
      - Larger head dimensions (c2 >= 24) for accurate direct box regression (no DFL)
      - No feature detachment for one2one head (backbone trains on both branches)
      - Softplus on ltrb predictions to prevent invalid negative distances
      - NMS-free post-processing (top-k only) for export; NMS applied by validator during eval
    """

    def __init__(self, nc: int = 80, reg_max: int = 1, end2end: bool = True, ch: tuple = ()):
        """Initialize LightEdgeDetect with improved head capacity for direct regression."""
        super().__init__(nc, reg_max, end2end, ch)
        c2 = max(24, ch[0] // 6, min(self.reg_max * 4, 24))
        c3 = max(ch[0] // 2, min(self.nc, 80))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        if self.reg_max <= 1:
            self.dfl = nn.Identity()
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Apply softplus to box predictions for stable training (no negative ltrb)."""
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat(
            [F.softplus(box_head[i](x[i])).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1
        )
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward without feature detachment for one2one — backbone trains on both branches."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            one2one = self.forward_head(x, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def bias_init(self):
        """Initialize biases with improved prior for high-res P2 features."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):
                a[-1].bias.data[:] = 2.0
                b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[i]) ** 2)
