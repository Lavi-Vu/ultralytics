# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""LightEdge-YOLO: Lightweight NMS-free object detection with RepViTGhost hybrid backbone."""

from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv, DWConv, GhostConv
from ultralytics.utils.ops import make_divisible

# ---------------------------------------------------------------------------
# Configurations — Nano (~2.8M params) and Small (~8.5M params)
# ---------------------------------------------------------------------------

LIGHTEDGE_CFG = {
    "nano": {
        "stem_channels": 16,
        "stage_configs": [
            {"channels": 32, "depth": 2, "stride": 2, "expand": 2},
            {"channels": 64, "depth": 3, "stride": 2, "expand": 2},
            {"channels": 128, "depth": 4, "stride": 2, "expand": 2},
            {"channels": 256, "depth": 2, "stride": 2, "expand": 2},
        ],
        "bifpn_channels": 96,
        "head_hidden": 64,
    },
    "small": {
        "stem_channels": 24,
        "stage_configs": [
            {"channels": 48, "depth": 4, "stride": 2, "expand": 2},
            {"channels": 96, "depth": 6, "stride": 2, "expand": 2},
            {"channels": 192, "depth": 8, "stride": 2, "expand": 2},
            {"channels": 384, "depth": 4, "stride": 2, "expand": 2},
        ],
        "bifpn_channels": 160,
        "head_hidden": 80,
    },
}


# ---------------------------------------------------------------------------
# ECA — Efficient Channel Attention
# ---------------------------------------------------------------------------

class ECA(nn.Module):
    """Efficient Channel Attention via fast 1D convolution after GAP."""

    def __init__(self, channels: int, gamma: float = 2.0, beta: float = 1.0):
        super().__init__()
        t = int(abs((math.log2(channels) + beta) / gamma))
        self.kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, 1, c)
        y = self.conv(y).view(b, c, 1, 1)
        return x * self.sigmoid(y)


# ---------------------------------------------------------------------------
# CoordAtt — Coordinate Attention
# ---------------------------------------------------------------------------

class CoordAtt(nn.Module):
    """Coordinate Attention: decomposes 2D attention into two 1D attentions."""

    def __init__(self, inp: int, oup: int, reduction: int = 32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        b, c, h, w = x.shape
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_h * a_w


# ---------------------------------------------------------------------------
# RepDWConv — Reparameterized Depthwise Convolution
# ---------------------------------------------------------------------------

class RepDWConv(nn.Module):
    """Multi-branch depthwise conv during training, fused single-path at inference."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1):
        super().__init__()
        assert k == 3 and c1 == c2
        self.c1 = c1
        self.c2 = c2
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, groups=c1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c1, c2, 1, s, 0, groups=c1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.bn_identity = nn.BatchNorm2d(c2) if s == 1 else None
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn1(self.conv1(x)) + self.bn2(self.conv2(x))
        if self.bn_identity is not None:
            y = y + self.bn_identity(x)
        return self.act(y)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse_convs(self):
        if not hasattr(self, "conv1"):
            return
        kernel1, bias1 = self._fuse_bn(self.conv1, self.bn1)
        kernel2, bias2 = self._fuse_bn(self.conv2, self.bn2)
        kernel2 = F.pad(kernel2, [1, 1, 1, 1])
        final_w, final_b = kernel1 + kernel2, bias1 + bias2
        if self.bn_identity is not None:
            id_w, id_b = self._fuse_bn(None, self.bn_identity)
            final_w, final_b = final_w + id_w, final_b + id_b
        self.conv = nn.Conv2d(
            self.c1, self.c2, 3, self.conv1.stride, 1,
            groups=self.c1, bias=True,
        ).requires_grad_(False)
        self.conv.weight.data.copy_(final_w)
        self.conv.bias.data.copy_(final_b)
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        self.__delattr__("bn1")
        self.__delattr__("bn2")
        if hasattr(self, "bn_identity"):
            self.__delattr__("bn_identity")
        self.forward = self.forward_fuse

    @staticmethod
    def _fuse_bn(conv: nn.Conv2d | None, bn: nn.BatchNorm2d):
        if conv is None:
            c = bn.weight.shape[0]
            id_k = torch.zeros(c, 1, 3, 3, device=bn.weight.device)
            id_k[:, :, 1, 1] = 1
            gamma, beta = bn.weight, bn.bias
            mean, var, eps = bn.running_mean, bn.running_var, bn.eps
            std = (var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return id_k * t, beta - mean * gamma / std
        gamma, beta = bn.weight, bn.bias
        mean, var, eps = bn.running_mean, bn.running_var, bn.eps
        std = (var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return conv.weight * t, beta - mean * gamma / std


# ---------------------------------------------------------------------------
# SPPFLight — lightweight SPPF
# ---------------------------------------------------------------------------

class SPPFLight(nn.Module):
    """SPPF with smaller kernel for multi-scale context with minimal params."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat((x, y1, y2, y3), 1))


# ---------------------------------------------------------------------------
# RepViTGhostBlock — core backbone building block
# ---------------------------------------------------------------------------

class RepViTGhostBlock(nn.Module):
    """Hybrid block: GhostConv → RepDWConv → CoordAtt → ECA → Proj + Skip.

    Design:
      - GhostConv as cheap expansion (half params vs standard conv)
      - RepDWConv for reparameterized depthwise processing
      - CoordAtt for spatial attention
      - ECA for channel attention
      - Depthwise separable design benefits quantization stability
    """

    def __init__(self, c_in: int, c_out: int, expand_ratio: float = 2.0):
        super().__init__()
        c_mid = make_divisible(int(c_in * expand_ratio), 8)
        self.ghost_conv = GhostConv(c_in, c_mid, k=1, s=1)
        self.rep_dw = RepDWConv(c_mid, c_mid, k=3, s=1)
        self.ca = CoordAtt(c_mid, c_mid)
        self.eca = ECA(c_mid)
        self.proj = Conv(c_mid, c_out, k=1, s=1, act=False)
        self.has_skip = c_in == c_out
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.ghost_conv(x)
        x = self.rep_dw(x)
        x = self.ca(x)
        x = self.eca(x)
        x = self.proj(x)
        if self.has_skip:
            x = x + identity
        return self.act(x)


# ---------------------------------------------------------------------------
# BiFPN Weighted Fusion
# ---------------------------------------------------------------------------

class BiFPNWeightedFusion(nn.Module):
    """Learnable weighted feature fusion — fast normalized fusion."""

    def __init__(self, num_inputs: int, channels: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(num_inputs, dtype=torch.float))
        self.eps = 1e-4
        self.conv = Conv(channels, channels, k=3, s=1)

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        y = sum(wi * xi for wi, xi in zip(w, xs))
        return self.conv(y)


# ---------------------------------------------------------------------------
# Adaptive BiFPN Neck  (P3, P4, P5 levels)
# ---------------------------------------------------------------------------

class AdaptiveBiFPN(nn.Module):
    """Top-down + bottom-up weighted BiFPN with learnable fusion weights."""

    def __init__(self, in_channels: list[int], bifpn_channels: int):
        super().__init__()
        c3, c4, c5 = in_channels
        ch = bifpn_channels

        self.proj_c3 = Conv(c3, ch, 1) if c3 != ch else nn.Identity()
        self.proj_c4 = Conv(c4, ch, 1) if c4 != ch else nn.Identity()
        self.proj_c5 = Conv(c5, ch, 1) if c5 != ch else nn.Identity()

        self.td_fuse4 = BiFPNWeightedFusion(2, ch)
        self.td_fuse3 = BiFPNWeightedFusion(2, ch)
        self.td_conv4 = Conv(ch, ch, 3)
        self.td_conv3 = Conv(ch, ch, 3)

        self.bu_fuse4 = BiFPNWeightedFusion(3, ch)
        self.bu_fuse5 = BiFPNWeightedFusion(3, ch)
        self.bu_conv4 = Conv(ch, ch, 3)
        self.bu_conv5 = Conv(ch, ch, 3)

    def forward(self, xs: list[torch.Tensor]) -> list[torch.Tensor]:
        p3, p4, p5 = xs
        p3 = self.proj_c3(p3)
        p4 = self.proj_c4(p4)
        p5 = self.proj_c5(p5)

        _, _, h3, w3 = p3.shape
        _, _, h4, w4 = p4.shape
        _, _, h5, w5 = p5.shape

        p5_td = p5
        p4_td = self.td_fuse4([p4, F.interpolate(p5_td, size=(h4, w4), mode="nearest")])
        p4_td = self.td_conv4(p4_td)
        p3_td = self.td_fuse3([p3, F.interpolate(p4_td, size=(h3, w3), mode="nearest")])
        p3_td = self.td_conv3(p3_td)

        p3_out = p3_td
        p4_out = self.bu_fuse4([p4, p4_td, F.interpolate(p3_out, size=(h4, w4), mode="nearest")])
        p4_out = self.bu_conv4(p4_out)
        p5_out = self.bu_fuse5([p5, p5_td, F.interpolate(p4_out, size=(h5, w5), mode="nearest")])
        p5_out = self.bu_conv5(p5_out)

        return [p3_out, p4_out, p5_out]


# ---------------------------------------------------------------------------
# LightEdge Head — NMS-free decoupled detection head
# ---------------------------------------------------------------------------

class LightEdgeHead(nn.Module):
    """NMS-free decoupled detection head with one-to-one matching.

    Design:
      - Decoupled cls/reg branches per FPN level
      - Direct ltrb box regression (no DFL)
      - One-to-one label assignment (STAL-inspired)
      - NMS-free via one-to-one matching + top-k selection

    Forward modes:
      - training: returns {"one2many": ..., "one2one": ...} dict
      - inference: returns (decoded_tensor, preds_dict) or decoded only (export)
    """

    dynamic = False
    export = False
    format = None
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    agnostic_nms = False
    xyxy = False
    end2end = True
    legacy = False

    def __init__(self, nc: int = 80, ch: tuple = (), hidden: int = 64):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.stride = torch.zeros(self.nl)

        self.box_head = nn.ModuleList(
            nn.Sequential(
                Conv(c, hidden, 3), Conv(hidden, hidden, 3),
                nn.Conv2d(hidden, 4, 1),
            ) for c in ch
        )
        self.cls_head = nn.ModuleList(
            nn.Sequential(
                Conv(c, hidden, 3), Conv(hidden, hidden, 3),
                nn.Conv2d(hidden, self.nc, 1),
            ) for c in ch
        )
        self.one2one_box_head = deepcopy(self.box_head)
        self.one2one_cls_head = deepcopy(self.cls_head)

    def forward_head(
        self, x: list[torch.Tensor],
        box_head: nn.ModuleList | None = None,
        cls_head: nn.ModuleList | None = None,
    ) -> dict:
        if box_head is None or cls_head is None:
            return dict()
        bs = x[0].shape[0]
        boxes = torch.cat([
            bh(f).view(bs, 4, -1) for bh, f in zip(box_head, x)
        ], dim=-1)
        scores = torch.cat([
            ch(f).view(bs, self.nc, -1) for ch, f in zip(cls_head, x)
        ], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(self, x: list[torch.Tensor]) -> dict | torch.Tensor | tuple:
        preds_o2m = self.forward_head(x, self.box_head, self.cls_head)
        x_detach = [xi.detach() for xi in x]
        preds_o2o = self.forward_head(x_detach, self.one2one_box_head, self.one2one_cls_head)
        preds = {"one2many": preds_o2m, "one2one": preds_o2o}

        if self.training:
            return preds

        y = self._inference(preds["one2one"])
        y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict) -> torch.Tensor:
        shape = x["feats"][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1).contiguous()
                                          for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape
        dbox = decode_bboxes(x["boxes"], self.anchors.unsqueeze(0), dim=1)
        dbox = dbox * self.strides
        cls = x["scores"].sigmoid()
        return torch.cat((dbox, cls), dim=1)

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        boxes, scores = preds.split([4, self.nc], dim=-1)
        k = self.max_det if self.export else min(self.max_det, scores.shape[1])
        ori_idx = scores.max(dim=-1)[0].topk(k, dim=1)[1].unsqueeze(-1)
        scores_g = scores.gather(dim=1, index=ori_idx.repeat(1, 1, self.nc))
        scores_flat, idx = scores_g.flatten(1).topk(k)
        scores_flat = scores_flat[..., None]
        label = (idx % self.nc)[..., None].float()
        box_idx = ori_idx[torch.arange(scores_flat.shape[0])[..., None], idx // self.nc]
        boxes_g = boxes.gather(dim=1, index=box_idx.repeat(1, 1, 4))
        return torch.cat([boxes_g, scores_flat, label], dim=-1)

    def bias_init(self):
        for a, b in zip(self.box_head, self.cls_head):
            a[-1].bias.data[:] = 2.0
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / 4)

    def fuse(self):
        self.box_head = self.cls_head = None

    def _apply(self, fn):
        self = super()._apply(fn)
        self.stride = fn(self.stride)
        self.anchors = fn(self.anchors)
        self.strides = fn(self.strides)
        return self


# ---------------------------------------------------------------------------
# Anchor utilities (standalone helpers for clean head implementation)
# ---------------------------------------------------------------------------

def make_anchors(feats: list[torch.Tensor], strides: torch.Tensor,
                 grid_cell_offset: float = 0.5):
    """Generate anchor points and stride tensor from feature maps."""
    anchor_points, stride_tensor = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, feat in enumerate(feats):
        _, _, h, w = feat.shape
        stride = strides[i]
        sx = torch.arange(w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def decode_bboxes(distance: torch.Tensor, anchor_points: torch.Tensor,
                  dim: int = 1) -> torch.Tensor:
    """Decode ltrb distances to xyxy boxes.

    Args:
        distance: (B, 4, N) ltrb distances from anchor points
        anchor_points: (B, 2, N) or (1, 2, N) anchor (cx, cy) coordinates
        dim: dimension to split along (1 for channel-first layout)
    Returns:
        (B, 4, N) xyxy boxes
    """
    lt, rb = distance.chunk(2, dim=dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    return torch.cat((x1y1, x2y2), dim=dim)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class VarifocalLoss(nn.Module):
    """Varifocal Loss: asymmetric focusing for dense detection."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred_score: torch.Tensor, gt_score: torch.Tensor,
                label: torch.Tensor) -> torch.Tensor:
        weight = self.alpha * pred_score.sigmoid().pow(self.gamma) * (1 - label) + gt_score * label
        loss = F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none")
        return (loss * weight).mean(1).sum()


class GIoULoss(nn.Module):
    """GIoU loss for box regression."""

    @staticmethod
    def forward(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_x1, pred_y1, pred_x2, pred_y2 = pred.unbind(1)
        tgt_x1, tgt_y1, tgt_x2, tgt_y2 = target.unbind(1)
        x1 = torch.min(pred_x1, tgt_x1)
        y1 = torch.min(pred_y1, tgt_y1)
        x2 = torch.max(pred_x2, tgt_x2)
        y2 = torch.max(pred_y2, tgt_y2)
        area_i = (torch.min(pred_x2, tgt_x2) - torch.max(pred_x1, tgt_x1)).clamp(min=0) * \
                 (torch.min(pred_y2, tgt_y2) - torch.max(pred_y1, tgt_y1)).clamp(min=0)
        a = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        b = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)
        iou = area_i / (a + b - area_i + 1e-7)
        area_c = ((x2 - x1).clamp(min=0)) * ((y2 - y1).clamp(min=0))
        giou = iou - (area_c - (a + b - area_i)) / (area_c + 1e-7)
        return (1 - giou).sum()


class SmallObjectBoostLoss(nn.Module):
    """Additional GIoU weight for small objects to improve AP_S."""

    def __init__(self, threshold: float = 32.0, boost_weight: float = 2.0):
        super().__init__()
        self.threshold = threshold
        self.boost = boost_weight

    def forward(self, giou: torch.Tensor, gt_boxes: torch.Tensor) -> torch.Tensor:
        area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        small = (area.sqrt() < self.threshold).float()
        return (giou * (1 + self.boost * small)).sum()


class LightEdgeLoss(nn.Module):
    """Combined loss: Varifocal + GIoU + SmallObjectBoost for NMS-free training.

    Uses TaskAligned matching:
      - one2one: best anchor per GT (for NMS-free inference)
      - one2many: top-k anchors per GT (auxiliary supervision)

    Compatible with Ultralytics trainer: __call__ returns (loss, loss_items) tuple.

    NOTE: does NOT store a reference to the model to avoid circular submodule recursion.
    """

    def __init__(self, nc: int):
        super().__init__()
        self.nc = nc
        self.vfl = VarifocalLoss()
        self.giou_fn = GIoULoss()
        self.obj_boost = SmallObjectBoostLoss()
        self.alpha = 0.5
        self.beta = 6.0

    def forward(self, preds: dict, batch: dict) -> tuple:
        """Compute loss. Returns (loss * batch_size, loss_items) tuple for trainer."""
        bs = batch["img"].shape[0]
        _, _, h, w = batch["img"].shape
        device = batch["img"].device
        head = preds["head"]

        gt_boxes = batch["bboxes"].to(device).clone()
        cx, cy, bw, bh = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2], gt_boxes[:, 3]
        gt_boxes[:, 0] = (cx - bw / 2) * w
        gt_boxes[:, 1] = (cy - bh / 2) * h
        gt_boxes[:, 2] = (cx + bw / 2) * w
        gt_boxes[:, 3] = (cy + bh / 2) * h

        gt_cls = batch["cls"].long().to(device).squeeze(-1)
        batch_idx = batch["batch_idx"].to(device)

        feats = preds["one2many"]["feats"]
        anchors, strides = make_anchors(feats, head.stride)
        anchors = anchors.unsqueeze(0)

        total_box = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)

        for branch, topk, w in [("one2one", 1, 1.0), ("one2many", 7, 0.5)]:
            p = preds.get(branch)
            if p is None:
                continue

            boxes_pred = p["boxes"]
            scores_pred = p["scores"]

            for b_idx in range(boxes_pred.shape[0]):
                mask = batch_idx == b_idx
                n_gt = mask.sum()
                if n_gt == 0:
                    continue

                gt_b = gt_boxes[mask]
                cls_b = gt_cls[mask]

                boxes_b = boxes_pred[b_idx:b_idx+1]
                scores_b = scores_pred[b_idx:b_idx+1, :, :]

                dbox = decode_bboxes(boxes_b, anchors.transpose(1, 2).contiguous(), dim=1)
                dbox = dbox * strides.t().unsqueeze(0)
                dbox = dbox.squeeze(0).permute(1, 0)

                iou = self._bbox_iou(dbox, gt_b)
                cls_scores = scores_b.squeeze(0).permute(1, 0).sigmoid()
                align = cls_scores[:, cls_b].pow(self.alpha) * iou.pow(self.beta)

                for j in range(n_gt):
                    _, topk_idx = align[:, j].topk(min(topk, align.shape[0]))
                    assign_scores = iou[topk_idx, j]

                    pos_boxes = dbox[topk_idx]
                    pos_scores = scores_b[:, :, topk_idx]

                    giou = self.giou_fn(pos_boxes, gt_b[j:j+1].expand_as(pos_boxes))
                    total_box = total_box + self.obj_boost(giou, gt_b[j:j+1].expand_as(pos_boxes)) * w

                    tgt = torch.zeros(1, self.nc, len(topk_idx), device=device)
                    tgt[0, cls_b[j]] = assign_scores
                    label = F.one_hot(cls_b[j].expand(len(topk_idx)),
                                      self.nc).float().permute(1, 0).unsqueeze(0)
                    total_cls = total_cls + self.vfl(pos_scores, tgt, label) * w

        loss = total_box + total_cls
        loss_items = torch.stack([total_box.detach(), total_cls.detach()])
        return loss * bs, loss_items

    @staticmethod
    def _bbox_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0])
        y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1])
        x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2])
        y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
        area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
        union = area1[:, None] + area2[None, :] - inter
        return inter / (union + 1e-7)


# ---------------------------------------------------------------------------
# Main LightEdge-YOLO Model
# ---------------------------------------------------------------------------

class LightEdgeYOLO(nn.Module):
    """LightEdge-YOLO: lightweight NMS-free object detector.

    Architecture:
      1. RepViTGhost hybrid backbone (4 stages, stride-4 P2 through stride-32 P5)
      2. Adaptive BiFPN neck (3-level weighted fusion)
      3. NMS-free decoupled detection head (one-to-one matching)

    Variants:
      - nano:   ~2.8M params, optimized for edge deployment
      - small:  ~8.5M params, higher accuracy for resource-constrained devices

    Usage:
        >>> model = LightEdgeYOLO("nano", nc=80)
        >>> out = model(torch.randn(1, 3, 640, 640))  # inference
        >>> model.train()
        >>> out = model(x)  # returns dict for loss computation
    """

    def __init__(self, variant: str = "nano", nc: int = 80, ch: int = 3, verbose: bool = True):
        super().__init__()
        self.variant = variant.lower()
        self.nc = nc
        self.stride = torch.tensor([8, 16, 32])
        self.names = {i: f"{i}" for i in range(nc)}
        self.args = None  # set by DetectionTrainer.set_model_attributes
        self.class_weights = None  # set by BaseTrainer.set_class_weights
        self.save = []
        self.yaml = None

        if self.variant in LIGHTEDGE_CFG:
            cfg = deepcopy(LIGHTEDGE_CFG[self.variant])
        else:
            from ultralytics.nn.tasks import yaml_model_load
            cfg = yaml_model_load(self.variant)
            cfg["stage_configs"] = [
                {"channels": c, "depth": d, "stride": s, "expand": e}
                for c, d, s, e in cfg["stage_configs"]
            ]
        self.cfg = cfg
        self.yaml = cfg

        self._build_backbone(ch)
        self._build_neck()
        self._build_head()
        self._init_weights()
        if verbose:
            self.info()

    # ---- Backbone ---------------------------------------------------------

    def _build_backbone(self, ch_in: int):
        cfg = self.cfg
        stem_ch = cfg["stem_channels"]
        stages_cfg = cfg["stage_configs"]

        backbone = []
        backbone.append(Conv(ch_in, stem_ch, k=3, s=2))

        stage_outs = []
        prev_ch = stem_ch
        for s_cfg in stages_cfg:
            out_ch = s_cfg["channels"]
            depth = s_cfg["depth"]
            stride = s_cfg["stride"]
            expand = s_cfg["expand"]

            backbone.append(Conv(prev_ch, out_ch, k=3, s=stride))
            cur_ch = out_ch
            for j in range(depth):
                block_in = cur_ch if j == 0 else out_ch
                backbone.append(RepViTGhostBlock(block_in, out_ch, expand))
                cur_ch = out_ch
            stage_outs.append(len(backbone) - 1)
            prev_ch = out_ch

        self._p2_idx, self._p3_idx, self._p4_idx, self._p5_idx = stage_outs
        self.backbone = nn.ModuleList(backbone)

    # ---- Neck -------------------------------------------------------------

    def _build_neck(self):
        ch = [self.cfg["stage_configs"][i]["channels"] for i in range(1, 4)]
        self.neck = AdaptiveBiFPN(ch, self.cfg["bifpn_channels"])
        self._neck_ch = self.cfg["bifpn_channels"]

    # ---- Head -------------------------------------------------------------

    def _build_head(self):
        self.head = LightEdgeHead(
            nc=self.nc,
            ch=(self._neck_ch,) * 3,
            hidden=self.cfg["head_hidden"],
        )
        self.head.stride = self.stride

    # ---- Init weights -----------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.zeros_(m.bias)
        self.head.bias_init()

    # ---- Forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor, *args, **kwargs) -> dict | torch.Tensor | tuple:
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor | tuple:
        feats = self._forward_backbone(x)
        neck_feats = self.neck(feats[1:])
        out = self.head(neck_feats)
        if isinstance(out, dict):
            out["head"] = self.head
        return out

    def _forward_backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        outs = []
        for m in self.backbone:
            x = m(x)
            outs.append(x)
        return [outs[self._p2_idx], outs[self._p3_idx],
                outs[self._p4_idx], outs[self._p5_idx]]

    def loss(self, batch: dict, preds=None) -> tuple:
        if not hasattr(self, "criterion"):
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(batch["img"])
        return self.criterion(preds, batch)

    def init_criterion(self):
        return LightEdgeLoss(nc=self.nc)

    # ---- Utilities --------------------------------------------------------

    def fuse(self, verbose: bool = True):
        from ultralytics.utils.torch_utils import fuse_conv_and_bn
        for m in self.modules():
            if isinstance(m, RepDWConv):
                m.fuse_convs()
            elif isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)
                delattr(m, "bn")
                m.forward = m.forward_fuse
        if isinstance(self.head, LightEdgeHead):
            self.head.fuse()
        if verbose and not self.is_fused:
            self.info(verbose=False)
        return self

    def info(self, detailed=False, verbose=True, imgsz=640):
        from ultralytics.utils.torch_utils import model_info
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def load(self, weights, verbose=True):
        from ultralytics.utils.torch_utils import intersect_dicts
        model = weights["model"] if isinstance(weights, dict) else weights
        csd = model.float().state_dict()
        csd = intersect_dicts(csd, self.state_dict())
        self.load_state_dict(csd, strict=False)
        if verbose:
            n = len(csd)
            print(f"Transferred {n}/{len(self.state_dict())} items")

    def is_fused(self):
        return sum(isinstance(m, nn.BatchNorm2d) for m in self.modules()) < 10

    def _apply(self, fn):
        self = super()._apply(fn)
        if hasattr(self, "head"):
            self.head = self.head._apply(fn)
        return self

    def __str__(self):
        n_p = sum(x.numel() for x in self.parameters())
        return (f"LightEdge-{self.variant.upper()}: "
                f"{n_p/1e6:.2f}M params, {self.nc} classes")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def lightedge_nano(nc: int = 80, verbose: bool = True) -> LightEdgeYOLO:
    return LightEdgeYOLO("nano", nc=nc, verbose=verbose)


def lightedge_small(nc: int = 80, verbose: bool = True) -> LightEdgeYOLO:
    return LightEdgeYOLO("small", nc=nc, verbose=verbose)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Export utilities
# ---------------------------------------------------------------------------


def export_onnx(model: LightEdgeYOLO, path: str = "lightedge.onnx",
                imgsz: int = 640, simplify: bool = True) -> str:
    """Export to ONNX format."""
    model.eval()
    model.head.export = True
    model.head.format = "onnx"
    x = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(
        model, x, path,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    if simplify:
        try:
            import onnx
            import onnxslim
            m = onnx.load(path)
            m = onnxslim.slim(m)
            onnx.save(m, path)
        except ImportError:
            pass
    model.head.export = False
    model.head.format = None
    return path


def export_tensorrt(model: LightEdgeYOLO, path: str = "lightedge.engine",
                    imgsz: int = 640, fp16: bool = True) -> str:
    """Export to TensorRT engine via ONNX intermediate."""
    onnx_path = path.replace(".engine", ".onnx")
    export_onnx(model, onnx_path, imgsz)
    import subprocess
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={path}",
        f"--minShapes=images:1x3x{imgsz}x{imgsz}",
        f"--optShapes=images:1x3x{imgsz}x{imgsz}",
        f"--maxShapes=images:16x3x{imgsz}x{imgsz}",
    ]
    if fp16:
        cmd.append("--fp16")
    subprocess.run(cmd, check=True)
    return path


def export_coreml(model: LightEdgeYOLO, path: str = "lightedge.mlpackage",
                  imgsz: int = 640) -> str:
    """Export to CoreML format."""
    model.eval()
    model.head.export = True
    model.head.format = "coreml"
    model.head.xyxy = True
    x = torch.randn(1, 3, imgsz, imgsz)
    traced = torch.jit.trace(model, x)
    import coremltools as ct
    ct_model = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="images", shape=(1, 3, imgsz, imgsz),
                              scale=1.0 / 255.0, bias=[0, 0, 0])],
    )
    ct_model.save(path)
    model.head.export = False
    model.head.format = None
    model.head.xyxy = False
    return path


def export_tflite(model: LightEdgeYOLO, path: str = "lightedge.tflite",
                  imgsz: int = 640, int8: bool = False) -> str:
    """Export to TFLite via ONNX-TF bridge."""
    import onnx
    from onnx_tf.backend import prepare
    onnx_path = path.replace(".tflite", ".onnx")
    export_onnx(model, onnx_path, imgsz)
    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    saved_model_dir = path.replace(".tflite", "_savedmodel")
    tf_rep.export_graph(saved_model_dir)
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if int8:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    return path


def export(model: LightEdgeYOLO, fmt: str = "onnx", imgsz: int = 640, **kwargs) -> str:
    """Convenience export to any supported format."""
    exports = {
        "onnx": export_onnx,
        "tensorrt": export_tensorrt,
        "coreml": export_coreml,
        "tflite": export_tflite,
    }
    if fmt not in exports:
        raise ValueError(f"Unsupported format: {fmt}. Choose from {list(exports.keys())}")
    return exports[fmt](model, path=f"lightedge_{model.variant}.{fmt}", imgsz=imgsz, **kwargs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LightEdge-YOLO")
    parser.add_argument("--variant", type=str, default="nano", choices=["nano", "small"])
    parser.add_argument("--nc", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--profile", action="store_true", help="Benchmark inference speed")
    parser.add_argument("--export", type=str, default=None,
                        choices=["onnx", "tensorrt", "coreml", "tflite"],
                        help="Export format")
    args = parser.parse_args()

    model = LightEdgeYOLO(args.variant, nc=args.nc)

    if args.profile:
        x = torch.randn(args.batch, 3, args.imgsz, args.imgsz)
        model.eval()
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        import time
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            _ = model(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = (time.time() - t0) / 100 * 1000
        print(f"Mean inference time: {t:.2f}ms")

    if args.export:
        path = export(model, args.export, imgsz=args.imgsz)
        print(f"Exported to {path}")

    x = torch.randn(args.batch, 3, args.imgsz, args.imgsz)
    model.eval()
    with torch.no_grad():
        out = model(x)
    print(f"Output: {type(out).__name__}", end="")
    if isinstance(out, tuple):
        print(f" pred={out[0].shape}, dict keys={list(out[1].keys())}")
    print(f"\nTotal params: {sum(p.numel() for p in model.parameters())/1e6:.3f}M")
