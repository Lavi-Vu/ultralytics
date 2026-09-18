#!/usr/bin/env python3
"""
High-quality Grad-CAM + Dense Feature visualization for LightEdgeDet.

Outputs per image (stem):
  {stem}_gradcam_{class_name}.jpg   — per-class Grad-CAM overlay at attention stage
  {stem}_features_P{level}.jpg      — top-K channel grid per backbone stage
  {stem}_neck_before_P{level}.jpg   — neck features before CrossScaleFusion
  {stem}_neck_after_P{level}.jpg    — neck features after CrossScaleFusion
  {stem}_gradcam_composite.jpg      — all Grad-CAM overlays in one panel
  {stem}_features_composite.jpg     — all backbone feature grids side by side
  {stem}_neck_composite.jpg         — before/after fusion comparison
  {stem}_composite.jpg              — combined (all sections stacked)

Usage:
    python scripts/visualize.py \
        --weights best_LED_small_6.pt \
        --source ultralytics/assets/bus.jpg \
        --device cpu --conf 0.25
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO

BG = (20, 20, 20)
FG = (255, 255, 255)


# ═══════════════════════════════════════════════════════════════════════════
#  Arguments
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="High-quality Grad-CAM + dense feature visualization")
    p.add_argument("--weights", required=True, help="path to trained .pt")
    p.add_argument("--source", required=True, help="image file or directory")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--top-channels", type=int, default=6,
                   help="channels per stage in dense feature grid")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--cell-size", type=int, default=256,
                   help="per-cell size for Grad-CAM overlays")
    p.add_argument("--tile-size", type=int, default=160,
                   help="per-channel tile size in feature grids")
    p.add_argument("--gap", type=int, default=16,
                   help="pixel gap between cells")
    p.add_argument("--stage", type=int, default=None,
                   help="override backbone stage index for Grad-CAM")
    p.add_argument("--output", default="viz_output")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  Preprocessing
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(image, imgsz, device):
    h, w = image.shape[:2]
    s = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dx, dy = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0


# ═══════════════════════════════════════════════════════════════════════════
#  Detection
# ═══════════════════════════════════════════════════════════════════════════

def run_detections(yolo, imgsz, conf, device):
    r = yolo.predict(imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return (r.boxes.xyxy.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
            r.boxes.cls.cpu().numpy().astype(int))


# ═══════════════════════════════════════════════════════════════════════════
#  Grad-CAM
# ═══════════════════════════════════════════════════════════════════════════

class _Hook:
    def __init__(self, mod):
        self.a = self.g = None
        self._hf = mod.register_forward_hook(self._fwd)
        self._hb = mod.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.a = o.detach() if torch.is_tensor(o) else o[0].detach()

    def _bwd(self, m, gi, go):
        self.g = go[0].detach()

    def remove(self):
        self._hf.remove()
        self._hb.remove()


def compute_gradcam(model, inp, stage_idx, class_id):
    """Grad-CAM at backbone.stages[stage_idx] for one class.

    Uses sum of positive raw logits as target — stable, dense gradients.
    """
    hook = _Hook(model.backbone.stages[stage_idx])
    x = inp.detach().requires_grad_(True)
    model.zero_grad()

    feat = model.backbone(x)
    feat = model.neck(feat)
    preds = model.detect.forward_head(feat, **model.detect.one2many)
    target = preds['scores'][0, class_id].clamp(min=0).sum()
    target.backward()

    feat, grad = hook.a, hook.g
    hook.remove()
    if feat is None or grad is None:
        raise RuntimeError(f"No gradient at backbone.stages.{stage_idx}")

    w = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((w * feat).sum(dim=1, keepdim=True))
    cam = cam - cam.min()
    mx = cam.max()
    if mx > 0:
        cam = cam / mx
    return cam[0, 0].cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
#  Neck feature capture (before/after CrossScaleFusion)
# ═══════════════════════════════════════════════════════════════════════════

def capture_neck_features(model, inp):
    """Capture neck features before and after the CrossScaleFusion gate.

    Returns (pre_fusion, post_fusion, gate_values):
      - pre_fusion: list of 4 tensors [P3, P4, P5, P6] before fusion
      - post_fusion: list of 4 tensors [P3, P4, P5, P6] after fusion
      - gate_values: numpy array of per-channel gate scalars (from Sigmoid)
    """
    neck = model.neck
    gate_capture = {}

    def _capture_gate(m, i, o):
        gate_capture['val'] = o.detach()

    hook = None
    if hasattr(neck, 'fusion') and hasattr(neck.fusion, 'gate'):
        hook = neck.fusion.gate.register_forward_hook(_capture_gate)

    with torch.no_grad():
        backbone_feats = model.backbone(inp)

        # Post-fusion: run full neck
        post_fusion = [f.detach() for f in model.neck(backbone_feats)]

    if hook is not None:
        hook.remove()

    # Extract gate values: [1, C, 1, 1] → [C]
    gate_values = None
    if 'val' in gate_capture:
        gate_values = gate_capture['val'][0, :, 0, 0].cpu().numpy()

    # Pre-fusion: manually run lateral → FPN → PAN, skip fusion
    with torch.no_grad():
        n = neck.num_levels
        reduced = [lat(f) for lat, f in zip(neck.lateral_convs, backbone_feats)]

        # Top-down (FPN)
        laterals = [neck.fpn_convs[-1](reduced[-1])]
        for i in range(n - 2, -1, -1):
            up = F.interpolate(laterals[-1], size=reduced[i].shape[2:],
                               mode="nearest")
            laterals.append(neck.fpn_convs[i](reduced[i] + up))
        laterals = laterals[::-1]

        # Bottom-up (PAN)
        pre = [neck.pan_convs[0](laterals[0])]
        for i in range(1, n):
            down = neck.down_convs[i - 1](pre[-1])
            pre.append(neck.pan_convs[i](laterals[i] + down))

        pre_fusion = [f.detach() for f in pre]

    return pre_fusion, post_fusion, gate_values


# ═══════════════════════════════════════════════════════════════════════════
#  Rendering helpers
# ═══════════════════════════════════════════════════════════════════════════

def _colormap(cam, cmap=cv2.COLORMAP_INFERNO):
    return cv2.applyColorMap(np.clip(cam * 255, 0, 255).astype(np.uint8), cmap)


def _overlay(bgr, cam, alpha=0.45, cmap=cv2.COLORMAP_INFERNO):
    hm = cv2.resize(_colormap(cam, cmap), (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_CUBIC)
    return cv2.addWeighted(bgr, 1 - alpha, hm, alpha, 0)


def _tile(ch, size, cmap=cv2.COLORMAP_JET):
    c = ch - ch.min()
    mx = c.max()
    if mx > 0:
        c = c / mx
    heat = cv2.applyColorMap(np.clip(c * 255, 0, 255).astype(np.uint8), cmap)
    return cv2.resize(heat, (size, size), interpolation=cv2.INTER_CUBIC)


def _ratio_tile(pre_ch, post_ch, size):
    """Render spatial difference (post - pre) with JET colormap."""
    diff = post_ch - pre_ch
    mx = max(np.abs(diff).max(), 1e-6)
    normalized = (diff / mx + 1) / 2  # map [-mx, mx] → [0, 1]
    heat = cv2.applyColorMap(np.clip(normalized * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_JET)
    return cv2.resize(heat, (size, size), interpolation=cv2.INTER_CUBIC)


def _gate_bar(gate_values, width, height):
    """Render gate values as a horizontal bar chart with JET colors."""
    n = len(gate_values)
    bar_h = max(1, (height - 40) // n)
    chart = np.full((height, width, 3), BG, dtype=np.uint8)

    for i, v in enumerate(gate_values):
        y = 40 + i * bar_h
        bar_w = int(v * (width - 60))
        # JET colormap: map gate value (0-1) to color
        color_val = np.clip(v * 255, 0, 255).astype(np.uint8)
        color_map = cv2.applyColorMap(color_val.reshape(1, 1), cv2.COLORMAP_JET)
        color = tuple(int(c) for c in color_map[0, 0])
        cv2.rectangle(chart, (40, y), (40 + bar_w, y + bar_h - 1), color, -1)

    return chart


def _box(img, box, label, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, FG, 1, cv2.LINE_AA)
    return img


def _title(img, text, h=40, bg=BG, fg=FG, scale=0.6, thick=1):
    bar = np.full((h, img.shape[1], 3), bg, dtype=np.uint8)
    cv2.putText(bar, text, (12, h // 2 + 6),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)
    return np.vstack([bar, img])


def _colorbar(w, h, cmap=cv2.COLORMAP_INFERNO):
    grad = np.linspace(255, 0, h).astype(np.uint8).reshape(-1, 1)
    bar = cv2.applyColorMap(grad, cmap)
    bar = cv2.resize(bar, (w, h), interpolation=cv2.INTER_CUBIC)
    bar = cv2.copyMakeBorder(bar, 0, 0, 0, 32, cv2.BORDER_CONSTANT, value=BG)
    for v, label in [(1.0, "1.0"), (0.5, "0.5"), (0.0, "0.0")]:
        y = int((1 - v) * h) + 6
        y = max(12, min(y, h + 18))
        cv2.putText(bar, label, (w + 4, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, FG, 1, cv2.LINE_AA)
    return bar


def _grid(cells, cols, gap=16, bg=BG):
    if not cells:
        return np.full((1, 1, 3), bg, dtype=np.uint8)

    max_h = max(c.shape[0] for c in cells)
    max_w = max(c.shape[1] for c in cells)

    normed = []
    for c in cells:
        ph = max_h - c.shape[0]
        pw = max_w - c.shape[1]
        if ph > 0 or pw > 0:
            c = cv2.copyMakeBorder(c, 0, ph, 0, pw,
                                   cv2.BORDER_CONSTANT, value=bg)
        normed.append(c)

    rows = (len(normed) + cols - 1) // cols
    while len(normed) < rows * cols:
        normed.append(np.full((max_h, max_w, 3), bg, dtype=np.uint8))

    row_imgs = []
    for r in range(rows):
        row = normed[r * cols:(r + 1) * cols]
        img = row[0]
        for c in row[1:]:
            gap_h = np.full((img.shape[0], gap, 3), bg, dtype=np.uint8)
            img = np.hstack([img, gap_h, c])
        row_imgs.append(img)

    result = row_imgs[0]
    for row in row_imgs[1:]:
        gap_v = np.full((gap, result.shape[1], 3), bg, dtype=np.uint8)
        result = np.vstack([result, gap_v, row])

    return result


def _pad_w(img, target_w, bg=BG):
    if img.shape[1] < target_w:
        return cv2.copyMakeBorder(img, 0, 0, 0, target_w - img.shape[1],
                                  cv2.BORDER_CONSTANT, value=bg)
    return img


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = torch.device(args.device if "cuda" in args.device
                          and torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()
    model.requires_grad_(True)

    nc = model.yaml.get('nc', 80)
    names = getattr(model, 'names', None) or {}
    def cls_name(c):
        if isinstance(names, dict):
            return names.get(c, f"cls{c}")
        return f"cls{c}"

    attn = model.yaml.get('backbone_attn_stages', [])
    channels = model.yaml.get('backbone_channels', [])

    # Pick target stage: deepest attention stage with usable resolution
    if args.stage is not None:
        stage_idx = args.stage
    else:
        candidates = [i for i, a in enumerate(attn) if a == 1]
        res_fn = lambda i: args.imgsz // (2 ** (i + 1))
        usable = [i for i in candidates if res_fn(i) >= 20]
        stage_idx = max(usable) if usable else max(candidates) if candidates else 0

    stage_res = args.imgsz // (2 ** (stage_idx + 1))
    print(f"Grad-CAM target: backbone.stages[{stage_idx}] "
          f"(P{stage_idx + 2}, {stage_res}x{stage_res}, "
          f"{channels[stage_idx]}ch)")

    # Source images
    src = Path(args.source)
    if src.is_file():
        images = [src]
    else:
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        images = sorted(f for f in src.rglob('*') if f.suffix.lower() in exts)

    gap = args.gap
    cs = args.cell_size
    ts = args.tile_size

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"  [skip] {img_path.name}")
            continue
        stem = img_path.stem
        inp = preprocess(bgr, args.imgsz, device)

        # ── Detections ──
        boxes, scores, labels = run_detections(yolo, args.imgsz, args.conf, device)

        # Highest-conf box per class
        reps = {}
        order = np.argsort(-scores)
        for idx in order:
            lb = int(labels[idx])
            if lb not in reps:
                reps[lb] = (boxes[idx], float(scores[idx]))

        det_img = bgr.copy()
        for lb, (bx, sc) in reps.items():
            _box(det_img, bx, f"{cls_name(lb)} {sc:.2f}")

        # ── Grad-CAM ──
        gc_cells = []
        gc_cells.append(_title(cv2.resize(bgr, (cs, cs),
                               interpolation=cv2.INTER_CUBIC), "Input"))
        gc_cells.append(_title(cv2.resize(det_img, (cs, cs),
                               interpolation=cv2.INTER_CUBIC), "Detections"))

        for lb, (bx, sc) in reps.items():
            try:
                cam = compute_gradcam(model, inp, stage_idx, lb)
                ov = _overlay(bgr, cam, args.alpha)
                ov = _box(ov, bx, f"{cls_name(lb)} {sc:.2f}")
                ov = cv2.resize(ov, (cs, cs), interpolation=cv2.INTER_CUBIC)
                # Save individual
                cv2.imwrite(str(out / f"{stem}_gradcam_{cls_name(lb)}.jpg"),
                            _title(ov, f"Grad-CAM P{stage_idx+2} "
                                       f"\u00b7 {cls_name(lb)}"))
                gc_cells.append(_title(ov, f"Grad-CAM \u00b7 {cls_name(lb)}"))
                print(f"  {stem}: Grad-CAM {cls_name(lb)} "
                      f"(conf {sc:.2f}) \u2713")
            except Exception as e:
                print(f"  {stem}: Grad-CAM {cls_name(lb)} \u2717 ({e})")

        # Color bar
        cb = _title(_colorbar(40, cs), "Scale")
        gc_cells.append(cb)

        gc_comp = _grid(gc_cells, cols=len(gc_cells), gap=gap)
        cv2.imwrite(str(out / f"{stem}_gradcam_composite.jpg"), gc_comp)
        print(f"  {stem}: gradcam composite \u2713")

        # ── Dense features ──
        with torch.no_grad():
            backbone_feats = model.backbone(inp)

        # backbone_feats: indices 0–3 correspond to P3–P6
        stage_labels = ["P3", "P4", "P5", "P6"]
        stage_sizes = [80, 40, 20, 10]  # for 640 input

        feat_cells = []
        for si, label in enumerate(stage_labels):
            if si >= len(backbone_feats):
                break
            feat = backbone_feats[si][0]  # [C, H, W]
            c, h, w = feat.shape
            l2 = feat.pow(2).sum(dim=(1, 2))
            n_show = min(args.top_channels, c)
            _, topk_idx = l2.topk(n_show)
            topk_idx = topk_idx.tolist()

            tiles = []
            for ci in topk_idx:
                t = _tile(feat[ci].cpu().numpy(), ts)
                # Channel label with background pill
                lbl = f"ch{ci}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.4, 1)
                # cv2.rectangle(t, (2, 2), (tw + 6, th + 6), (0, 0, 0), -1)
                # cv2.putText(t, lbl, (4, th + 4),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, FG, 1,
                #             cv2.LINE_AA)
                tiles.append(t)

            # Arrange in 2 rows
            cols_feat = (n_show + 1) // 2
            cell = _grid(tiles, cols=cols_feat, gap=4)
            cell = _title(cell, f"{label} ({h}x{w}, {c}ch)")
            feat_cells.append(cell)

            # Save individual stage
            cv2.imwrite(str(out / f"{stem}_features_{label}.jpg"), cell)

        feat_comp = _grid(feat_cells, cols=4, gap=gap)
        feat_comp = _title(feat_comp, "Dense Features (Backbone)")
        cv2.imwrite(str(out / f"{stem}_features_composite.jpg"), feat_comp)
        print(f"  {stem}: features composite \u2713")

        # ── Neck features (CrossScaleFusion gate effect) ──
        pre_fusion, post_fusion, gate_values = capture_neck_features(model, inp)

        neck_cells = []

        # Gate bar chart (one per level, but values are shared across levels)
        if gate_values is not None:
            gate_chart = _gate_bar(gate_values, width=ts * 3, height=ts * 2 + 40)
            gate_chart = _title(gate_chart,
                                f"Gate Values ({len(gate_values)}ch, "
                                f"mean={gate_values.mean():.3f})")
            neck_cells.append(gate_chart)

        for si, label in enumerate(stage_labels):
            if si >= len(pre_fusion) or si >= len(post_fusion):
                break

            pre_feat = pre_fusion[si][0]   # [C, H, W]
            post_feat = post_fusion[si][0]
            c, h, w = pre_feat.shape

            # Top-K channels from post-fusion (by L2 magnitude)
            l2 = post_feat.pow(2).sum(dim=(1, 2))
            n_show = min(args.top_channels, c)
            _, topk_idx = l2.topk(n_show)
            topk_idx = topk_idx.tolist()

            # Render spatial difference tiles (post - pre)
            tiles = []
            for ci in topk_idx:
                pre_ch = pre_feat[ci].cpu().numpy()
                post_ch = post_feat[ci].cpu().numpy()
                t = _ratio_tile(pre_ch, post_ch, ts)
                # Channel label with background pill
                lbl = f"ch{ci}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                              0.4, 1)
                # cv2.rectangle(t, (2, 2), (tw + 6, th + 6), (0, 0, 0), -1)
                # cv2.putText(t, lbl, (4, th + 4),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.4, FG, 1,
                #             cv2.LINE_AA)
                tiles.append(t)

            # Arrange in 2 rows
            cols_feat = (n_show + 1) // 2
            cell = _grid(tiles, cols=cols_feat, gap=4)
            cell = _title(cell, f"{label} ({h}x{w}, {c}ch)")
            neck_cells.append(cell)

            # Save individual: before, after, and difference
            pre_tiles = [_tile(pre_feat[ci].cpu().numpy(), ts)
                         for ci in topk_idx]
            post_tiles = [_tile(post_feat[ci].cpu().numpy(), ts)
                          for ci in topk_idx]
            diff_tiles = [_ratio_tile(pre_feat[ci].cpu().numpy(),
                                      post_feat[ci].cpu().numpy(), ts)
                          for ci in topk_idx]
            cv2.imwrite(str(out / f"{stem}_neck_before_{label}.jpg"),
                        _title(_grid(pre_tiles, cols=cols_feat, gap=4),
                               f"{label} Before Fusion"))
            cv2.imwrite(str(out / f"{stem}_neck_after_{label}.jpg"),
                        _title(_grid(post_tiles, cols=cols_feat, gap=4),
                               f"{label} After Fusion"))
            cv2.imwrite(str(out / f"{stem}_neck_diff_{label}.jpg"),
                        _title(_grid(diff_tiles, cols=cols_feat, gap=4),
                               f"{label} Difference (post-pre)"))

        neck_comp = _grid(neck_cells, cols=len(neck_cells), gap=gap)
        neck_comp = _title(neck_comp,
                           "CrossScaleFusion Gate Effect (blue=suppressed, red=amplified)")
        cv2.imwrite(str(out / f"{stem}_neck_composite.jpg"), neck_comp)
        print(f"  {stem}: neck composite \u2713")

        # ── Combined ──
        mw = max(gc_comp.shape[1], feat_comp.shape[1], neck_comp.shape[1])
        combined = np.vstack([
            _pad_w(gc_comp, mw),
            np.full((gap, mw, 3), BG, dtype=np.uint8),
            _pad_w(feat_comp, mw),
            np.full((gap, mw, 3), BG, dtype=np.uint8),
            _pad_w(neck_comp, mw)])
        cv2.imwrite(str(out / f"{stem}_composite.jpg"), combined)
        print(f"  {stem}: combined \u2713\n")

    print("Done.")


if __name__ == "__main__":
    main()
