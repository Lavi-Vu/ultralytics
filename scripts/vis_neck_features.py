#!/usr/bin/env python3
"""
Dense feature-map visualization for LightEdgeDet.

Visualizes the CNN/depthwise feature channels at each neck level (P3-P6),
both *before* and *after* the CrossScaleFusion global gate:

    top row    = neck output BEFORE the CrossScaleFusion gate
    bottom row = neck output AFTER  the CrossScaleFusion gate

For each (level, before/after) cell we select the top-K channels by L2
magnitude and render them as JET heatmaps, arranged as a grid:

    ┌──────────┬──────────┬──────────┬──────────┐
    │ P3-before │ P4-before │ P5-before │ P6-before │   <- pre-fusion
    ├──────────┼──────────┼──────────┼──────────┤
    │ P3-after  │ P4-after  │ P5-after  │ P6-after  │   <- post-fusion
    └──────────┴──────────┴──────────┴──────────┘

The per-channel grids are upsampled to a common display size so the whole
figure is a clean 2×4 montage.

Usage:
    python scripts/vis_neck_features.py \
        --weights /path/to/best.pt \
        --source bus.jpg \
        --output neck_features \
        --device cpu \
        --top-channels 8 \
        --cell-size 160

Outputs
    neck_features/{stem}_neck_features.jpg   (full 2×4 montage)
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Dense feature-map viz at neck levels P3-P6 "
                    "before/after CrossScaleFusion gate"
    )
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--top-channels", type=int, default=8,
                        help="channels shown per (level, before/after) cell")
    parser.add_argument("--cell-size", type=int, default=160,
                        help="per-channel tile display size (square)")
    parser.add_argument("--output", type=str, default="neck_features")
    parser.add_argument("--save-individual", action="store_true",
                        help="also save each level's before/after grid")
    return parser.parse_args()


# ============================================================
# PREPROCESS (letterbox + normalize)
# ============================================================

def preprocess(image, imgsz, device):
    h, w = image.shape[:2]
    scale = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    dw, dh = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return tensor.to(device)


# ============================================================
# CAPTURE pre/post fusion features
# ============================================================

def capture_neck_features(model, inp):
    """Run forward and return (pre_fusion, post_fusion).

    ``pre_fusion``  : list of 4 tensors, neck output BEFORE the gate
    ``post_fusion`` : list of 4 tensors, neck output AFTER  the gate
    """
    neck = model.neck

    with torch.no_grad():
        backbone_feats = model.backbone(inp)
        neck_out = model.neck(backbone_feats)
        post_fusion = [f.detach() for f in neck_out]

        # Re-run the neck internals up to (but excluding) fusion to get the
        # pre-gate features: lateral -> top-down FPN -> bottom-up PAN.
        reduced = [lat(f) for lat, f in zip(neck.lateral_convs, backbone_feats)]
        n = neck.num_levels
        laterals = [neck.fpn_convs[-1](reduced[-1])]
        for i in range(n - 2, -1, -1):
            up = torch.nn.functional.interpolate(
                laterals[-1], size=reduced[i].shape[2:], mode="nearest")
            laterals.append(neck.fpn_convs[i](reduced[i] + up))
        laterals = laterals[::-1]
        pre = [neck.pan_convs[0](laterals[0])]
        for i in range(1, n):
            down = neck.down_convs[i - 1](pre[-1])
            pre.append(neck.pan_convs[i](laterals[i] + down))
        pre_fusion = [f.detach() for f in pre]

    return pre_fusion, post_fusion


# ============================================================
# RENDER a single channel as a JET heatmap tile
# ============================================================

def channel_tile(ch_map, size, label=None):
    ch = ch_map - ch_map.min()
    mx = ch.max()
    if mx > 0:
        ch = ch / mx
    ch_uint8 = np.uint8(ch * 255)
    heat = cv2.applyColorMap(ch_uint8, cv2.COLORMAP_JET)
    heat = cv2.resize(heat, (size, size), interpolation=cv2.INTER_LINEAR)
    if label is not None:
        cv2.putText(heat, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return heat


# ============================================================
# RENDER one (level, before/after) cell as a channel grid
# ============================================================

def render_cell(feat, top_k, size, cols=4, title=None):
    """feat: [C, H, W] tensor -> grid of top-k channels."""
    c, h, w = feat.shape
    l2 = feat.pow(2).sum(dim=(1, 2))
    n_show = min(top_k, c)
    _, idx = l2.topk(n_show)
    idx = idx.tolist()

    rows = (n_show + cols - 1) // cols
    grid = np.full((rows * size, cols * size, 3), 20, dtype=np.uint8)
    for i, ci in enumerate(idx):
        r, col = divmod(i, cols)
        grid[r * size:(r + 1) * size, col * size:(col + 1) * size] = \
            channel_tile(feat[ci].cpu().numpy(), size, label=f"ch{ci}")

    if title is not None:
        bar = np.full((28, grid.shape[1], 3), 0, dtype=np.uint8)
        cv2.putText(bar, f"{title}  [{c} ch, {h}x{w}]", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255),
                    1, cv2.LINE_AA)
        grid = np.vstack([bar, grid])
    return grid


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- device ----
    device = torch.device(args.device if args.device.startswith("cuda")
                          and torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ---- model ----
    print("[INFO] Loading model...")
    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()

    # ---- image ----
    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(args.source)
    stem = Path(args.source).stem
    inp = preprocess(image, args.imgsz, device)

    # ---- capture ----
    pre_fusion, post_fusion = capture_neck_features(model, inp)

    levels = ["P3", "P4", "P5", "P6"]
    size = args.cell_size
    top_k = args.top_channels
    cols = 4  # one column per P-level

    # ---- build each cell ----
    top_grids = []  # before
    bot_grids = []  # after
    for lv, pre_f, post_f in zip(levels, pre_fusion, post_fusion):
        if args.save_individual:
            pre_grid = render_cell(pre_f[0], top_k, size, title=f"{lv} BEFORE gate")
            post_grid = render_cell(post_f[0], top_k, size, title=f"{lv} AFTER gate")
            cv2.imwrite(str(out_dir / f"{stem}_neck_{lv}_before.jpg"), pre_grid)
            cv2.imwrite(str(out_dir / f"{stem}_neck_{lv}_after.jpg"), post_grid)
        # strip title for montage cells
        top_grids.append(render_cell(pre_f[0], top_k, size, cols=cols))
        bot_grids.append(render_cell(post_f[0], top_k, size, cols=cols))

    # ---- equalize widths across cells then concatenate rows ----
    def pad_row(cells, target_w):
        out = []
        for g in cells:
            if g.shape[1] < target_w:
                pad = np.full((g.shape[0], target_w - g.shape[1], 3),
                              20, dtype=np.uint8)
                g = np.hstack([g, pad])
            out.append(g)
        return np.hstack(out)

    max_w = max([g.shape[1] for g in top_grids + bot_grids])
    top_row = pad_row(top_grids, max_w)
    bot_row = pad_row(bot_grids, max_w)

    # --- row labels (strip) ---
    def add_row_label(grid, text):
        bar = np.full((30, grid.shape[1], 3), 0, dtype=np.uint8)
        cv2.putText(bar, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return np.vstack([bar, grid])

    top_row = add_row_label(top_row, "BEFORE CrossScaleFusion gate")
    bot_row = add_row_label(bot_row, "AFTER  CrossScaleFusion gate")

    montage = np.vstack([top_row, bot_row])

    # column headers
    header = np.full((30, montage.shape[1], 3), 0, dtype=np.uint8)
    x = 8
    for lv in levels:
        cv2.putText(header, lv, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        x += max_w + 4
    montage = np.vstack([header, montage])

    out_path = out_dir / f"{stem}_neck_features.jpg"
    cv2.imwrite(str(out_path), montage)

    print(f"[INFO] Saved → {out_path}")

    # ---- print level/channel stats ----
    for lv, pre_f, post_f in zip(levels, pre_fusion, post_fusion):
        pre_n, post_n = pre_f[0].norm().item(), post_f[0].norm().item()
        print(f"  {lv}: pre {tuple(pre_f[0].shape)} "
              f"(L2={pre_n:.1f})  |  post {tuple(post_f[0].shape)} "
              f"(L2={post_n:.1f})")


if __name__ == "__main__":
    main()
