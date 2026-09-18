#!/usr/bin/env python3
"""
Saliency maps (Grad-CAM-style overlays) at the deepest attention-augmented
backbone stage, computed for representative detections.

For each confident detection (per class) the script back-propagates a
class-specific target score from the detection head and produces a Grad-CAM
overlay at the deepest stage that contains a LitePSA attention module.  This
shows which spatial regions of the attention-augmented feature map the network
relies on to localise each detected object.

The target stage defaults to the deepest attention-augmented backbone stage
(the last stage index where ``attn_stages[i] == 1``) that still has usable
spatial resolution (>= 20x20 at the input size), i.e. P5 for the standard
configs.  This can be overridden with ``--stage` (e.g. ``4`` for P6).

For every image:

    {stem}_preds.jpg                     # input with detections drawn
    {stem}_sal_P5_class{c}_overlay.jpg   # Grad-CAM overlay per detection
    {stem}_sal_P5_composite.jpg          # input + all detection overlays

Usage:
    python scripts/vis_attention_saliency.py \
        --weights /path/to/best.pt \
        --source bus.jpg \
        --device cpu \
        --conf 0.25
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics import YOLO


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Grad-CAM saliency at the deepest attention backbone stage")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--source", type=str, required=True,
                   help="image file or directory")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--conf", type=float, default=0.25,
                   help="min conf for a 'representative' detection")
    p.add_argument("--stage", type=int, default=None,
                   help="override target backbone stage index "
                        "(default: auto deepest attention stage)")
    p.add_argument("--min-res", type=int, default=20,
                   help="ignore attention stages smaller than this (px)")
    p.add_argument("--top-detections", type=int, default=10,
                   help="max detections to visualize per image")
    p.add_argument("--alpha", type=float, default=0.45)
    p.add_argument("--cols", type=int, default=3,
                   help="columns per row in the composite grid")
    p.add_argument("--output", type=str, default="attn_saliency")
    p.add_argument("--names", type=str, default=None,
                   help="path to class names file (txt, one per line)")
    return p.parse_args()


# ============================================================
# PREPROCESS / DETECTION HELPERS
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
    return torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0


def raw_head_output(model, inp):
    """backbone -> neck -> forward_head; return preds dict with raw logits."""
    feat = model.backbone(inp)
    feat = model.neck(feat)
    return model.detect.forward_head(feat, **model.detect.one2many)


def run_detections(yolo, imgsz, conf, device):
    """Run YOLO.predict (full decode + NMS) and return representative detections.

    Returns (boxes[N,4], scores[N], labels[N]) in original-image coords.
    """
    r = yolo.predict(imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)
    boxes = r.boxes.xyxy.cpu().numpy()
    scores = r.boxes.conf.cpu().numpy()
    labels = r.boxes.cls.cpu().numpy().astype(int)
    return boxes, scores, labels


# ============================================================
# Grad-CAM at a backbone stage
# ============================================================

class _Hook:
    """Capture forward activation + backward gradient for a module."""
    def __init__(self, mod):
        self.act = self.grad = None
        self._hf = mod.register_forward_hook(self._on_fwd)
        self._hb = mod.register_full_backward_hook(self._on_bwd)

    def _on_fwd(self, m, i, o):
        self.act = o.detach() if torch.is_tensor(o) else o[0].detach()

    def _on_bwd(self, m, gi, go):
        self.grad = go[0].detach()

    def remove(self):
        self._hf.remove()
        self._hb.remove()


def gradcam_at_stage(model, inp, stage_idx, class_id):
    """Grad-CAM at the given backbone stage for a target class.

    Uses the sum of positive raw class logits over all anchors as the target
    score (dense, stable gradients) — matches the working saliency recipes.
    """
    mod = model.backbone.stages[stage_idx]
    hook = _Hook(mod)
    x = inp.detach().requires_grad_(True)
    model.zero_grad()

    preds = raw_head_output(model, x)
    scores = preds['scores']                 # [1, nc, N] raw logits
    target = scores[0, class_id].clamp(min=0).sum()
    target.backward()

    feat, grad = hook.act, hook.grad
    hook.remove()
    if feat is None or grad is None:
        raise RuntimeError(f"No gradient captured at backbone.stages.{stage_idx}")

    w = grad.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
    cam = F.relu((w * feat).sum(dim=1, keepdim=True))  # (1, 1, H, W)
    cam = cam - cam.min()
    m = cam.max()
    if m > 0:
        cam = cam / m
    return cam[0, 0].cpu().numpy()                   # (H, W) in [0, 1]


# ============================================================
# Visualization
# ============================================================

def heatmap(cam):
    return cv2.applyColorMap(np.clip(cam * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def overlay(bgr, cam, alpha):
    hm = heatmap(cam)
    hm = cv2.resize(hm, (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(bgr, 1 - alpha, hm, alpha, 0)


def draw_box(img, box, label, color=(0, 255, 0)):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(img, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
    return img


def add_title(img, text):
    bar = np.zeros((34, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    device = torch.device(args.device if args.device.startswith("cuda")
                          and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    yolo = YOLO(args.weights)
    model = yolo.model.to(device).eval()
    model.requires_grad_(True)   # need gradients for Grad-CAM

    nc = model.yaml.get('nc', 80)
    # class names from the model where available
    names = getattr(model, 'names', None) or (
        yolo.model.names if hasattr(yolo.model, 'names') else None)
    if not names:
        names = {i: f"cls{i}" for i in range(nc)}
    def cls_name(c):
        n = names.get(c, names[c] if c in names else None)
        return f"cls{c}" if n is None else (n if isinstance(n, str) else str(n))

    attn = model.yaml.get('backbone_attn_stages', [])
    channels = list(model.yaml.get('backbone_channels', []))

    # ---- pick target stage: deepest attention stage with usable res ----
    if args.stage is not None:
        stage_idx = args.stage
    else:
        candidates = [i for i, a in enumerate(attn) if a == 1]
        # feature-map size at stage i (input imgsz / 2^(i+1))
        def res(i):
            return args.imgsz // (2 ** (i + 1))
        stage_usable = [i for i in candidates if res(i) >= args.min_res]
        if not stage_usable:
            stage_usable = candidates
        stage_idx = max(stage_usable)   # deepest usable attention stage
    stage_res = args.imgsz // (2 ** (stage_idx + 1))
    print(f"[INFO] Attention-augmented stages: {attn}")
    print(f"[INFO] Target stage: backbone.stages.{stage_idx} "
          f"(res {stage_res}x{stage_res}, channels {channels[stage_idx] if stage_idx < len(channels) else '?'})")

    # ---- sources ----
    src = Path(args.source)
    if src.is_file():
        images = [src]
    elif src.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        images = sorted(f for f in src.rglob('*') if f.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f"Not found: {src}")

    for img_path in images:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"[skip] {img_path.name}")
            continue
        stem = img_path.stem
        inp = preprocess(bgr, args.imgsz, device)

        # ---- detections ----
        boxes, scores, labels = run_detections(yolo, args.imgsz, args.conf, device)
        n_show = min(len(boxes), args.top_detections)
        if n_show == 0:
            print(f"  {stem}: no detections above conf {args.conf}")
            continue

        # pick the single highest-confidence box per detected class
        reps = {}   # class_id -> (box, score)
        order = np.argsort(-scores)[:n_show]
        for idx in order:
            lb = int(labels[idx])
            if lb not in reps:
                reps[lb] = (boxes[idx], float(scores[idx]))

        annot = bgr.copy()
        for lb, (bx, sc) in reps.items():
            annot = draw_box(annot, bx, f"{cls_name(lb)} {sc:.2f}")
        cv2.imwrite(str(out_dir / f"{stem}_preds.jpg"), annot)

        # ---- one Grad-CAM saliency map per detected class ----
        panels = [(bgr.copy(), "Input")]
        for lb, (bx, sc) in reps.items():
            try:
                cam = gradcam_at_stage(model, inp, stage_idx, lb)
                ov = overlay(bgr, cam, args.alpha)
                ov = draw_box(ov, bx, f"{cls_name(lb)} {sc:.2f}")
                fname = (out_dir /
                         f"{stem}_sal_P{stage_idx+2}_class{lb}.jpg")
                cv2.imwrite(str(fname), add_title(
                    ov, f"Grad-CAM @ P{stage_idx+2} · {cls_name(lb)}"))
                panels.append((ov, cls_name(lb)))
                print(f"  {stem}: class {cls_name(lb)} (conf {sc:.2f}) ✓")
            except Exception as e:
                print(f"  {stem}: class {cls_name(lb)} ✗ ({e})")

        # ---- composite grid: one cell per class saliency map ----
        if len(panels) > 1:
            cols = max(1, args.cols)
            rows = (len(panels) + cols - 1) // cols
            while len(panels) < rows * cols:
                panels.append((np.full_like(panels[0][0], 20), ""))
            w = max(p[0].shape[1] for p in panels)
            h = max(p[0].shape[0] for p in panels)
            cells = []
            for im, lbl in panels:
                im = add_title(im, lbl)
                if im.shape[1] != w or im.shape[0] != h:
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)
                cells.append(im)
            grid_rows = []
            for r in range(rows):
                grid_rows.append(np.hstack(cells[r * cols:(r + 1) * cols]))
            comp = np.vstack(grid_rows)
            cv2.imwrite(str(out_dir / f"{stem}_sal_P{stage_idx+2}_composite.jpg"), comp)

    print("\nDone.")


if __name__ == "__main__":
    main()
