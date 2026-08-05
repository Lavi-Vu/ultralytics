"""
LightEdgeDet custom modules for ultralytics integration.

Provides the HybridBackbone (CNN+MViT) and LitePAFPN neck as ultralytics-
compatible modules.  Each module's __init__ follows the (c1, c2, ...) convention
expected by parse_model so it plugs directly into a YAML model definition.
"""

import copy
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import DFL, SPPF
from ultralytics.nn.modules.conv import Conv, DWConv, autopad
from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import fuse_conv_and_bn


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
#  RepDWConv — reparameterized depthwise conv (train multi-branch, fuse at inference)
# ---------------------------------------------------------------------------

class RepDWConv(nn.Module):
    """Reparameterized depthwise convolution.

    During training the block is a sum of a 3x3 depthwise branch, a 1x1
    depthwise branch and (when stride==1) an identity branch — more capacity for
    the same inference cost.  ``fuse()`` folds the branches into a single 3x3
    depthwise convolution, so inference/export runs at plain-depthwise FLOPs.
    """

    def __init__(self, c: int, k: int = 3, s: int = 1, p: int = None):
        """Initialize RepDWConv.

        Args:
            c (int): Number of channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
        """
        super().__init__()
        p = autopad(k, p)
        self.c, self.k, self.s, self.p = c, k, s, p
        self.conv3 = Conv(c, c, k, s, p, g=c, act=False)
        self.conv1 = Conv(c, c, 1, 1, 0, g=c, act=False) if s == 1 else None
        self.add_identity = s == 1
        self.fused = False

    def forward(self, x):
        y = self.conv3(x)
        if self.conv1 is not None:
            y = y + self.conv1(x)
        if self.add_identity:
            y = y + x
        return y

    def fuse(self):
        """Fold all branches into a single depthwise convolution (exact)."""
        if self.fused:
            return
        self.fused = True
        pad = self.k // 2
        conv = fuse_conv_and_bn(self.conv3.conv, self.conv3.bn)
        if self.conv1 is not None:
            conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
            w1 = torch.zeros_like(conv.weight)
            w1[..., pad, pad] = conv1.weight.squeeze(-1).squeeze(-1)
            conv.weight.data.add_(w1)
            conv.bias.data.add_(conv1.bias)
        if self.add_identity:
            wi = torch.zeros_like(conv.weight)
            wi[..., pad, pad] = 1.0
            conv.weight.data.add_(wi)
        self.conv3.conv = conv
        self.conv3.forward = self.conv3.forward_fuse
        self.conv3.__delattr__("bn")
        self.conv1 = None
        self.add_identity = False
        self.forward = self.forward_fuse

    def forward_fuse(self, x):
        return self.conv3(x)


# ---------------------------------------------------------------------------
#  LiteBlock — [1x1 expand] -> [DW 3x3 stride] -> [SE] -> [1x1 proj] + residual
# ---------------------------------------------------------------------------

class LiteBlock(nn.Module):
    """Depthwise-separable MBConv-style block.

    Constructor: (c1, c2, depth, expand_ratio, se_ratio, drop_path_rate, stride).
    ``depth`` is accepted for API uniformity but ignored (depth is encoded in
    the YAML repeat count).  With ``reparam=True`` the depthwise 3x3 is a
    RepDWConv, giving train-time multi-branch capacity at zero inference cost.
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
        reparam: bool = True,
    ):
        super().__init__()
        mid_c = int(c1 * expand_ratio)
        self.use_residual = stride == 1 and c1 == c2

        # 1x1 expansion
        self.expand = nn.Identity() if expand_ratio == 1.0 else Conv(c1, mid_c, 1)

        # Depthwise 3x3 (reparameterized when stride==1)
        if reparam and stride == 1:
            self.dw = RepDWConv(mid_c, 3, 1, 1)
        else:
            self.dw = Conv(mid_c, mid_c, 3, stride, 1, g=mid_c, act=False)

        # SE
        self.se = nn.Identity() if se_ratio == 0.0 else SqueezeExcitation(
            mid_c, reduction=max(int(1 / se_ratio), 4)
        )

        # 1x1 projection
        self.proj = Conv(mid_c, c2, 1, act=False)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        h = self.dw(self.expand(x))
        h = F.silu(h, inplace=True)
        h = self.proj(self.se(h))
        return self.drop_path(h) + x if self.use_residual else h

    def fuse(self):
        """Fuse the reparameterized depthwise conv (if present)."""
        if isinstance(self.dw, RepDWConv) and not self.dw.fused:
            self.dw.fuse()


# ---------------------------------------------------------------------------
#  LitePSA — depthwise-separable split-route attention block
# ---------------------------------------------------------------------------

class LitePSA(nn.Module):
    """Depthwise-separable convolutional attention block (C2PSA-inspired).

    Split-route design without any quadratic attention: one route preserves
    local spatial detail with a depthwise 3x3, the other carries global channel
    context through a depthwise conv gated by a squeeze-excitation signal.  All
    operations are depthwise-separable apart from the cheap 1x1 split/fuse.
    """

    def __init__(self, c: int, e: float = 0.5):
        """Initialize LitePSA.

        Args:
            c (int): Input and output channels.
            e (float): Hidden expansion ratio for the split-route.
        """
        super().__init__()
        c_ = max(int(c * e), 8)
        self.cv1 = Conv(c, 2 * c_, 1)
        self.local = nn.Sequential(Conv(c_, c_, 3, 1, 1, g=c_), Conv(c_, c_, 1, act=False))
        self.context = nn.Sequential(Conv(c_, c_, 3, 1, 1, g=c_), Conv(c_, c_, 1, act=False))
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c_, max(c_ // 4, 4), 1),
            Conv(max(c_ // 4, 4), c_, 1, act=False),
            nn.Sigmoid(),
        )
        self.cv2 = Conv(2 * c_, c, 1, act=False)

    def forward(self, x):
        a, b = self.cv1(x).split((self.cv1.conv.out_channels // 2,) * 2, dim=1)
        a = self.local(a)
        b = self.context(b)
        b = b * self.gate(b)
        return self.cv2(torch.cat((a, b), 1))


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
        use_sppf: bool = True,
        psa_ratio: float = 0.5,
        reparam: bool = True,
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
                    blocks.append(LitePSA(out_c, e=psa_ratio))
                else:
                    blocks.append(LiteBlock(in_c, out_c, 1, expand, se_r, dp, stride, reparam))
                in_c = out_c
            self.stages.append(nn.Sequential(*blocks))

        self.out_indices = [1, 2, 3, 4]
        self.channels_list = channels_list

        # Global-context SPPF on the deepest stage (P6) — cheap multi-scale pooling
        # that flows into every neck level via the FPN top-down path.
        self.sppf = SPPF(channels_list[-1], channels_list[-1], k=5) if use_sppf else nn.Identity()

    def forward(self, x):
        features = []
        x = self.stem(x)
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                features.append(x)
        if isinstance(self.sppf, SPPF):
            features[-1] = self.sppf(features[-1])
        return features


# ---------------------------------------------------------------------------
#  LitePAFPN — Lightweight PAFPN neck (list -> list)
# ---------------------------------------------------------------------------

class _LiteConvBlock(nn.Module):
    """Depthwise separable conv 3x3 with residual."""
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
        return self.conv(x) + x


class _LiteDownBlock(nn.Module):
    """Learnable stride-2 depthwise-separable downsampling."""
    def __init__(self, c, **kw):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, stride=2, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class CrossScaleFusion(nn.Module):
    """RT-DETRv4-inspired lightweight cross-scale feature fusion.

    Aggregates the neck's multi-scale outputs at the deepest resolution with a
    compact multi-head attention, then broadcasts the global context back to
    every level through a channel gate.  Gives all scales (including P3 small
    objects) multi-scale global context at negligible FLOP cost.
    """

    def __init__(self, c: int, num_levels: int = 4, num_heads: int = 4):
        super().__init__()
        self.cv = nn.Conv2d(num_levels * c, c, 1, bias=False)
        self.norm = nn.LayerNorm(c)
        while c % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.attn = nn.MultiheadAttention(c, num_heads, batch_first=True, bias=False)
        self.ffn = nn.Sequential(
            nn.Conv2d(c, c * 2, 1), nn.SiLU(inplace=True), nn.Conv2d(c * 2, c, 1, bias=False)
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c, c, 1), nn.SiLU(inplace=True), nn.Conv2d(c, c, 1), nn.Sigmoid()
        )

    def forward(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        target = feats[-1].shape[2:]
        stacked = torch.cat(
            [F.interpolate(f, size=target, mode="bilinear", align_corners=False) for f in feats], dim=1
        )
        fused = self.cv(stacked)
        B, C, H, W = fused.shape
        u = fused.permute(0, 2, 3, 1).reshape(B, H * W, C)
        u = self.attn(self.norm(u), self.norm(u), self.norm(u), need_weights=False)[0]
        u = u.reshape(B, H, W, C).permute(0, 3, 1, 2)
        fused = fused + self.ffn(fused + u)
        gate = self.gate(F.adaptive_avg_pool2d(fused, 1))
        return [f * gate for f in feats]


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
        use_cross_fusion: bool = True,
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
        self.use_cross_fusion = use_cross_fusion

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
        self.fpn_convs = nn.ModuleList(
            [nn.Sequential(*[blk(c2) for _ in range(self.num_blocks)]) for _ in range(self.num_levels)]
        )
        self.pan_convs = nn.ModuleList(
            [nn.Sequential(*[blk(c2) for _ in range(self.num_blocks)]) for _ in range(self.num_levels)]
        )
        self.down_convs = nn.ModuleList([_LiteDownBlock(c2) for _ in range(self.num_levels - 1)])
        self.fusion = CrossScaleFusion(c2, self.num_levels) if self.use_cross_fusion else nn.Identity()

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
            down = self.down_convs[i - 1](outputs[-1])
            outputs.append(self.pan_convs[i](laterals[i] + down))

        # Cross-scale global-context fusion (RT-DETRv4-inspired)
        outputs = self.fusion(outputs)

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

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = (), head_channels: int = 56):
        # NOTE: super() is called with end2end=False so it does NOT deepcopy the default
        # per-level cv2/cv3 it would build before we replace them with our shared branches.
        super().__init__(nc=nc, reg_max=reg_max, end2end=False, ch=ch)
        self._end2end = bool(end2end)
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

        if self._end2end:
            # One-to-one (E2E) branches, detached in Detect.forward; same shared design.
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Channel-reduce each level, then apply the shared reg/cls branches."""
        if box_head is None or cls_head is None:  # fused inference (one2many removed)
            return dict()
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

        def _init(box_head, cls_head):
            reg_last = box_head[0][-1]
            cls_last = cls_head[0][-1]
            reg_last.bias.data[:] = 2.0  # box
            cls_last.bias.data[: self.nc] = math.log(5 / self.nc / (640 / self.stride[0]) ** 2)

        _init(self.cv2, self.cv3)
        if self._end2end:
            _init(self.one2one_cv2, self.one2one_cv3)
