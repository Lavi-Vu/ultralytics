"""
LightEdgeDet custom modules for ultralytics integration.

Provides the HybridBackbone (CNN+MViT) and LitePAFPN neck as ultralytics-
compatible modules.  Each module's __init__ follows the (c1, c2, ...) convention
expected by parse_model so it plugs directly into a YAML model definition.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv
from ultralytics.nn.modules.head import Detect


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(mid, channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = self.sigmoid(self.fc2(self.act(self.fc1(s))))
        return x * s


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(
            torch.full(shape, keep, device=x.device, dtype=x.dtype)
        )
        return x * mask / keep


# ---------------------------------------------------------------------------
#  LiteBlock — [1x1 expand] -> [DW 3x3 stride] -> [SE] -> [1x1 proj] + residual
# ---------------------------------------------------------------------------

class LiteBlock(nn.Module):
    """EfficientNet-style block compatible with parse_model.

    Constructor: (c1, c2, depth, expand_ratio, se_ratio, drop_path_rate)
    ``depth`` is accepted for API uniformity but ignored (depth is encoded in
    the YAML repeat count).
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        depth: int = 1,
        expand_ratio: float = 4.0,
        se_ratio: float = 0.0,
        drop_path_rate: float = 0.0,
        stride: int = 1,
    ):
        super().__init__()
        mid_c = int(c1 * expand_ratio)
        self.use_residual = stride == 1 and c1 == c2

        # 1x1 expansion
        self.expand = nn.Identity() if expand_ratio == 1.0 else nn.Sequential(
            nn.Conv2d(c1, mid_c, 1, bias=False),
            nn.BatchNorm2d(mid_c),
            nn.SiLU(inplace=True),
        )

        # Depthwise 3x3
        self.dw = nn.Sequential(
            nn.Conv2d(mid_c, mid_c, 3, stride=stride, padding=1, groups=mid_c, bias=False),
            nn.BatchNorm2d(mid_c),
        )

        # SE
        self.se = nn.Identity() if se_ratio == 0.0 else SqueezeExcitation(
            mid_c, reduction=max(int(1 / se_ratio), 4)
        )

        # 1x1 projection
        self.proj = nn.Sequential(
            nn.Conv2d(mid_c, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        h = self.proj(self.se(self.dw(self.expand(x))))
        return self.drop_path(h) + x if self.use_residual else h


# ---------------------------------------------------------------------------
#  MViTBlock — MobileViT-style inline transformer
# ---------------------------------------------------------------------------

class MViTBlock(nn.Module):
    """Local DW conv + global MHSA + FFN."""

    def __init__(
        self,
        channels: int,
        ff_hidden_dim: int = 384,
        num_heads: int = 4,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.local_rep = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.norm = nn.LayerNorm(channels)
        while channels % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ff_hidden_dim, bias=False),
            nn.LayerNorm(ff_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(ff_hidden_dim, channels, bias=False),
            nn.LayerNorm(channels),
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        local_feat = self.local_rep(x)
        u = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        u = self.norm(u)
        u = self.attn(u, u, u, need_weights=False)[0]
        u = self.ffn(u).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return self.drop_path(u) + local_feat


# ---------------------------------------------------------------------------
#  HybridBackbone — CNN + MViT, 5 stages, outputs [P3, P4, P5, P6]
# ---------------------------------------------------------------------------

class HybridBackbone(nn.Module):
    """Backbone compatible with ``parse_model``.

    Constructor signature matches base_modules convention:
    ``(c1, c2, *extra_args)`` where ``c2`` is kept for compatibility but the
    real output channels come from ``channels_list``.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        channels_list: str = "[24,48,96,144,192]",
        depths: str = "[2,3,4,3,2]",
        expand_ratios: str = "[4.0,4.0,4.0,4.0,4.0]",
        se_ratios: str = "[0.25,0.25,0.25,0.25,0.25]",
        attn_stages: str = "[0,0,0,1,1]",
        stem_channels: int = 16,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        import ast as _ast
        if isinstance(channels_list, str):
            channels_list = _ast.literal_eval(channels_list)
        if isinstance(depths, str):
            depths = _ast.literal_eval(depths)
        if isinstance(expand_ratios, str):
            expand_ratios = _ast.literal_eval(expand_ratios)
        if isinstance(se_ratios, str):
            se_ratios = _ast.literal_eval(se_ratios)
        if isinstance(attn_stages, str):
            attn_stages = _ast.literal_eval(attn_stages)

        self.stem = nn.Sequential(
            nn.Conv2d(c1, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.SiLU(inplace=True),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        dpr_idx = 0
        self.stages = nn.ModuleList()
        in_c = stem_channels

        for out_c, depth, expand, se_r, use_attn in zip(
            channels_list, depths, expand_ratios, se_ratios, attn_stages
        ):
            blocks = []
            for bi in range(depth):
                stride = 2 if bi == 0 else 1
                dp = dpr[dpr_idx]; dpr_idx += 1
                if use_attn and bi == depth // 2:
                    ff_dim = max(128, out_c * 2)
                    nh = max(2, out_c // 32)
                    blocks.append(MViTBlock(in_c, ff_dim, nh, dp))
                else:
                    blocks.append(LiteBlock(in_c, out_c, 1, expand, se_r, dp, stride))
                in_c = out_c
            self.stages.append(nn.Sequential(*blocks))

        self.out_indices = [1, 2, 3, 4]
        self.channels_list = channels_list

    def forward(self, x):
        features = []
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                features.append(x)
        return features


# ---------------------------------------------------------------------------
#  LitePAFPN — Lightweight PAFPN neck (list -> list)
# ---------------------------------------------------------------------------

class _LiteConvBlock(nn.Module):
    """Depthwise separable conv 3x3."""
    def __init__(self, c, c2=None, **kw):
        super().__init__()
        c2 = c2 or c
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, c2, 1, bias=False),
            nn.BatchNorm2d(c2), nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class LitePAFPN(nn.Module):
    """Lightweight PAFPN — takes a list of feature maps, returns a list.

    Constructor follows (c1, c2, ...) convention for parse_model.
    ``c1`` is ignored (real input channels inferred from backbone).
    ``c2`` is the output channel count for every level.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        num_blocks: int = 2,
        use_depthwise: bool = True,
    ):
        super().__init__()
        self.out_channels = c2
        self._out_channels_list = [c2, c2, c2, c2]  # 4 levels

        # We'll lazily build lateral + fpn + pan convs on first forward so
        # we know the actual input channel counts.
        self._built = False
        self._c_in = None
        self.num_levels = 4
        self.num_blocks = num_blocks
        self.use_depthwise = use_depthwise

    def _lazy_build(self, c_in_list):
        """Build sub-modules the first time we see real input channels."""
        if self._built:
            return
        self._built = True
        self._c_in = c_in_list
        c2 = self.out_channels
        blk = _LiteConvBlock if self.use_depthwise else lambda c, c2=None: nn.Sequential(
            nn.Conv2d(c, c2 or c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c2 or c), nn.SiLU(inplace=True),
        )

        self.lateral_convs = nn.ModuleList()
        for c in c_in_list:
            self.lateral_convs.append(
                nn.Sequential(nn.Conv2d(c, c2, 1), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
                if c != c2 else nn.Identity()
            )
        self.fpn_convs = nn.ModuleList([blk(c2) for _ in range(self.num_levels)])
        self.pan_convs = nn.ModuleList([blk(c2) for _ in range(self.num_levels)])

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        self._lazy_build([f.shape[1] for f in inputs])

        reduced = [lat(f) for lat, f in zip(self.lateral_convs, inputs)]

        # Top-down (FPN)
        laterals = [reduced[-1]]
        for i in range(self.num_levels - 2, -1, -1):
            up = F.interpolate(laterals[-1], size=reduced[i].shape[2:], mode="nearest")
            laterals.append(self.fpn_convs[i](reduced[i] + up))
        laterals = laterals[::-1]

        # Bottom-up (PAN)
        outputs = [laterals[0]]
        for i in range(1, self.num_levels):
            down = F.max_pool2d(outputs[-1], kernel_size=2, stride=2)
            outputs.append(self.pan_convs[i](laterals[i] + down))

        return outputs


class LightDetectHead(Detect):
    """Lightweight decoupled detection head for LightEdgeDet.

    Drops Detect's per-level 3x3 conv stacks in favor of a shared, depthwise-
    separable decoupled head (matching the original LightEdgeDet design):

    - per-level 1x1 channel reduce (neck -> head_channels)
    - ONE shared reg branch and ONE shared cls branch applied to every level
    - branches are depthwise + pointwise convs (cheap, no dense 3x3)

    Shares Detect's forward/forward_head/_inference machinery, so v8DetectionLoss,
    validation, and export all work unchanged.
    """

    def __init__(self, nc: int = 80, reg_max=16, ch: tuple = (), head_channels: int = 56):
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        c, c2 = ch[0], head_channels
        self.head_channels = c2

        # Per-level 1x1 channel reduce (neck -> head_channels), identity if already aligned.
        self.reduce = nn.ModuleList(
            nn.Sequential(nn.Conv2d(ci, c2, 1, bias=False), nn.BatchNorm2d(c2), nn.SiLU(inplace=True))
            if ci != c2
            else nn.Identity()
            for ci in ch
        )

        # Single shared branches (depthwise-separable), applied to every level.
        # cv2/cv3 are length-1 ModuleLists: ultralytics iterates them per level, but the
        # SAME module runs on every level. Keeping one entry (not nl copies of the same
        # object) stops thop/GFLOPs from counting the shared convs nl times.
        shared_reg = nn.Sequential(
            nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c2, 1)),
            nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c2, 1)),
            nn.Conv2d(c2, 4 * self.reg_max, 1),
        )
        shared_cls = nn.Sequential(
            nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c2, 1)),
            nn.Sequential(DWConv(c2, c2, 3), Conv(c2, c2, 1)),
            nn.Conv2d(c2, self.nc, 1),
        )
        self.cv2 = nn.ModuleList([shared_reg])
        self.cv3 = nn.ModuleList([shared_cls])

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Channel-reduce each level, then apply the shared reg/cls branches."""
        reduced = [self.reduce[i](xi) for i, xi in enumerate(x)]
        bs = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i if len(box_head) > 1 else 0](reduced[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)],
            dim=-1,
        )
        scores = torch.cat(
            [cls_head[i if len(cls_head) > 1 else 0](reduced[i]).view(bs, self.nc, -1) for i in range(self.nl)],
            dim=-1,
        )
        return dict(boxes=boxes, scores=scores, feats=x)

    def bias_init(self):
        """Initialize the shared head with a focal-loss prior (single value, no per-level split)."""
        reg_last = self.cv2[0][-1]
        cls_last = self.cv3[0][-1]
        reg_last.bias.data[:] = 2.0  # box
        cls_last.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[0]) ** 2)
