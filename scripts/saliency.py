#!/usr/bin/env python3
"""
Saliency Map Visualization for LightEdgeDet (ultralytics fork).

Computes gradient-based saliency maps showing which pixels drive the
model's highest-confidence prediction. Three methods:

  vanilla   — |grad · input|, raw but noisy
  smoothgrad — averaged vanilla over Gaussian-perturbed copies
  gradcam  — gradient-weighted feature map from a chosen layer

Usage:
    python ultralytics/scripts/saliency.py --source path/to/image.jpg
    python ultralytics/scripts/saliency.py --source imgs/ --model small --method gradcam
    python ultralytics/scripts/saliency.py --source img.jpg --model path/to/config.yaml
    python ultralytics/scripts/saliency.py --source img.jpg --model path/to/checkpoint.pt

Author: LightEdgeDet Team
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='LightEdgeDet saliency (ultralytics)')
    p.add_argument('--source', type=str, required=True,
                   help='Image file or directory')
    p.add_argument('--model', type=str, default='nano',
                   help='Model: preset name (nano/small/v2/medium/large), '
                        'path to a YAML config, or path to a .pt checkpoint')
    p.add_argument('--weights', type=str, default=None,
                   help='Optional checkpoint (.pt) to load on top of --model')
    p.add_argument('--img-size', type=int, default=640)
    p.add_argument('--method', type=str, default='all',
                   choices=['vanilla', 'smoothgrad', 'gradcam', 'all'])
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--save-dir', type=str, default='runs/saliency')
    p.add_argument('--target-layer', type=str, default=None,
                   help='Grad-CAM target(s). Comma-separated, each optionally '
                        'prefixed with "label:": e.g. '
                        '"P3:backbone.stages.1,P4:backbone.stages.2,P5:backbone.stages.3,'
                        'Fusion:neck.fusion"')
    p.add_argument('--smoothgrad-n', type=int, default=20,
                   help='Noise samples for smoothgrad')
    p.add_argument('--smoothgrad-sigma', type=float, default=0.1,
                   help='Noise std (relative to [0,1] range)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_PRESET_MAP = {
    'nano':  'lightedgedet_nano.yaml',
    'small': 'lightedgedet_small.yaml',
    'v2':    'lightedgedet_small_v2.yaml',
    'medium':'lightedgedet_medium.yaml',
    'large': 'lightedgedet_large.yaml',
}

_MODELS_DIR = Path(__file__).resolve().parent.parent / 'ultralytics' / 'cfg' / 'models'

# Default Grad-CAM targets (label, dot path)
_DEFAULT_TARGETS = [
    ('P3',       'backbone.stages.1'),
    ('P4',       'backbone.stages.2'),
    ('P5',       'backbone.stages.3'),
    ('Fusion',   'neck.fusion.cv'),
]


def parse_target_layers(raw: str | None):
    """Parse --target-layer string into [(label, path), ...].

    Accepts: "path" or "label:path", comma-separated.
    If None or empty, returns _DEFAULT_TARGETS.
    """
    if not raw:
        return list(_DEFAULT_TARGETS)
    targets = []
    for tok in raw.split(','):
        tok = tok.strip()
        if ':' in tok and not tok.startswith('backbone.') and not tok.startswith('neck.'):
            label, path = tok.split(':', 1)
        else:
            path = tok
            # auto-label from the last two path segments
            parts = path.split('.')
            label = '.'.join(parts[-2:]) if len(parts) >= 2 else parts[-1]
        targets.append((label.strip(), path.strip()))
    return targets


def load_model(name: str, weights: str | None, device: str):
    """Build and optionally load a LightEdgeDet model.

    ``name`` can be:
      - a preset key (nano / small / v2 / medium / large)
      - a path to a .yaml model config
      - a path to a .pt checkpoint (builds + loads in one step)
    """
    from ultralytics.nn.tasks import LightEdgeDetModel
    import yaml

    name_path = Path(name)

    # --- resolve YAML ------------------------------------------------------
    if name in _PRESET_MAP:
        yaml_path = _MODELS_DIR / _PRESET_MAP[name]
    elif name_path.suffix == '.yaml':
        yaml_path = name_path
    elif name_path.suffix == '.pt':
        # load checkpoint; model config is embedded in the .pt
        ckpt = torch.load(name_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, LightEdgeDetModel):
            model = ckpt.float().eval()
            if weights:
                _load_extra_weights(model, weights)
            model.to(device)
            return model
        # look for YAML config in checkpoint
        yaml_src = ckpt.get('yaml', None)
        if yaml_src is None:
            # ultralytics trainer stores the model path in train_args
            train_args = ckpt.get('train_args', {})
            model_path = train_args.get('model', None) if isinstance(train_args, dict) else None
            if model_path:
                model_file = Path(model_path)
                if not model_file.is_absolute():
                    # try relative to cwd first, then relative to cfg/models
                    if not model_file.exists():
                        model_file = _MODELS_DIR / model_file.name
                if model_file.exists():
                    yaml_src = yaml.safe_load(open(model_file))
        if yaml_src is None:
            raise ValueError(
                f'{name} is a .pt but contains no embedded YAML config. '
                'Pass a .yaml via --model and the .pt via --weights instead.'
            )
        cfg = yaml_src if isinstance(yaml_src, dict) else yaml.safe_load(yaml_src)
        model = LightEdgeDetModel(cfg, verbose=False)
        sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
        if isinstance(sd, LightEdgeDetModel):
            model = sd.float().eval()
        else:
            sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
            missing, _ = model.load_state_dict(sd, strict=False)
            print(f'  Loaded {name} ({len(missing)} missing keys)')
        if weights:
            _load_extra_weights(model, weights)
        model.eval().to(device)
        return model
    else:
        # try as a preset name with a helpful error
        avail = list(_PRESET_MAP.keys())
        raise ValueError(
            f'Unknown model: {name!r}. '
            f'Use a preset ({", ".join(avail)}), a .yaml path, or a .pt path.'
        )

    # --- build from YAML ---------------------------------------------------
    cfg = yaml.safe_load(open(yaml_path))
    model = LightEdgeDetModel(cfg, verbose=False)

    if weights:
        _load_extra_weights(model, weights)

    model.eval().to(device)
    return model


def _load_extra_weights(model, weights: str):
    """Load a .pt checkpoint into an already-built model."""
    if not Path(weights).exists():
        print(f'  [warn] weights not found: {weights}')
        return
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt.get('state_dict', ckpt))
    if isinstance(sd, torch.nn.Module):
        sd = sd.state_dict()
    sd = {k: v.float() if v.is_floating_point() else v for k, v in sd.items()}
    missing, _ = model.load_state_dict(sd, strict=False)
    print(f'  Loaded {weights} ({len(missing)} missing keys)')


# ---------------------------------------------------------------------------
# Preprocessing — matches ultralytics training (/255, no mean/std)
# ---------------------------------------------------------------------------

def preprocess(rgb: np.ndarray, size: int):
    """RGB uint8 (H,W,3) → (1,3,size,size) tensor, letterbox + /255."""
    h0, w0 = rgb.shape[:2]
    r = min(size / w0, size / h0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    t = torch.from_numpy(canvas).permute(2, 0, 1).float().div_(255.0)
    return t.unsqueeze(0), r


# ---------------------------------------------------------------------------
# Forward helpers
# ---------------------------------------------------------------------------

def _raw_output(model, inp):
    """Run backbone → neck → head; return raw class logits [1, 80, N]."""
    feat = model.backbone(inp)
    feat = model.neck(feat)
    preds = model.detect.forward_head(feat, **model.detect.one2many)
    return preds['scores']  # [1, nc, num_anchors] raw logits, no decode


def _max_cls_score(raw):
    """Scalar target: max class logit across all anchors."""
    return raw[0].max()


# ---------------------------------------------------------------------------
# Saliency methods
# ---------------------------------------------------------------------------

def vanilla_saliency(model, inp, sigma=0.0, n=1):
    """|grad · input|, optionally averaged over n noisy copies."""
    acc = torch.zeros(inp.shape[2:])  # (H, W)

    for i in range(max(1, n)):
        x = inp.clone().detach()
        if sigma > 0 and i > 0:
            x = x + torch.randn_like(x) * sigma
        x.requires_grad_(True)

        raw = _raw_output(model, x)
        _max_cls_score(raw).backward()
        acc += (x.detach() * x.grad)[0].abs().mean(dim=0)
        model.zero_grad()

    return acc / max(1, n)  # (H, W)


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------

class _Hook:
    def __init__(self, mod):
        self.fwd = None
        self.bwd = None
        self._h1 = mod.register_forward_hook(self._on_fwd)
        self._h2 = mod.register_full_backward_hook(self._on_bwd)

    def _on_fwd(self, m, i, o):
        if isinstance(o, (list, tuple)):
            self.fwd = [t.detach() if torch.is_tensor(t) else t for t in o]
        else:
            self.fwd = o.detach()

    def _on_bwd(self, m, gi, go):
        self.bwd = go[0].detach()

    def remove(self):
        self._h1.remove()
        self._h2.remove()


def gradcam_saliency(model, inp, target_path='backbone.stages.4'):
    """Grad-CAM over the module at `target_path`."""
    parts = target_path.split('.')
    mod = model
    for p in parts:
        mod = getattr(mod, p)

    hook = _Hook(mod)
    x = inp.detach().requires_grad_(True)
    raw = _raw_output(model, x)
    _max_cls_score(raw).backward()

    feat, grad = hook.fwd, hook.bwd
    hook.remove()

    if feat is None or grad is None:
        raise RuntimeError(f'Hook captured nothing from {target_path}')

    # handle list outputs (e.g. CrossScaleFusion returns [tensor, ...])
    if isinstance(feat, list):
        # use the first tensor (P3-level, highest res)
        feat = feat[0]
    if isinstance(grad, list):
        grad = grad[0]

    # global-average-pool gradients → channel importance weights
    w = grad.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
    cam = F.relu((w * feat).sum(dim=1, keepdim=True))  # (1, 1, H, W)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    return cam[0, 0].cpu()  # (H, W)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _heatmap(hw):
    return cv2.applyColorMap(np.clip(hw * 255, 0, 255).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def overlay(bgr, cam, alpha=0.45):
    hm = _heatmap(cam)
    hm = cv2.resize(hm, (bgr.shape[1], bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(bgr, 1 - alpha, hm, alpha, 0)


def save(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[warn] CUDA unavailable, falling back to CPU')
        args.device = 'cpu'

    print('=' * 60)
    label = Path(args.model).name if '/' in args.model or args.model.endswith(('.yaml', '.pt')) else args.model
    print(f'LightEdgeDet saliency  ({label}, {args.method})')
    print('=' * 60)

    model = load_model(args.model, args.weights, args.device)

    src = Path(args.source)
    if src.is_file():
        images = [src]
    elif src.is_dir():
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
        images = sorted(f for f in src.rglob('*') if f.suffix.lower() in exts)
    else:
        raise FileNotFoundError(f'Not found: {src}')

    do_v  = args.method in ('vanilla', 'all')
    do_sg = args.method in ('smoothgrad', 'all')
    do_gc = args.method in ('gradcam', 'all')
    targets = parse_target_layers(args.target_layer)

    print(f'  {len(images)} image(s) | {args.img_size}px | {args.device}')
    if do_gc:
        print(f'  gradcam targets: {", ".join(t[0] for t in targets)}')
    print('-' * 60)

    for path in images:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f'  [skip] {path.name}')
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inp, _ = preprocess(rgb, args.img_size)
        inp = inp.to(args.device)
        stem = path.stem

        # ---- compute each method once, cache overlays ----
        overlays = []  # [(label, overlay_bgr)]
        overlays.append(('Input', bgr.copy()))

        if do_v:
            sal = vanilla_saliency(model, inp).cpu().numpy()
            save(Path(args.save_dir) / f'{stem}_vanilla.jpg', overlay(bgr, sal))
            overlays.append(('Vanilla', overlay(bgr, sal)))
            print(f'  {stem}  vanilla ✓')

        if do_sg:
            sal = vanilla_saliency(model, inp,
                                   sigma=args.smoothgrad_sigma,
                                   n=args.smoothgrad_n).cpu().numpy()
            save(Path(args.save_dir) / f'{stem}_smoothgrad.jpg', overlay(bgr, sal))
            overlays.append(('SmoothGrad', overlay(bgr, sal)))
            print(f'  {stem}  smoothgrad ✓')

        # Grad-CAM: per-target individual saves
        for label, path_str in targets:
            try:
                cam = gradcam_saliency(model, inp, path_str).numpy()
                save(Path(args.save_dir) / f'{stem}_gradcam_{label}.jpg',
                     overlay(bgr, cam))
                overlays.append((f'Grad-CAM {label}', overlay(bgr, cam)))
                print(f'  {stem}  gradcam {label} ✓')
            except Exception as e:
                print(f'  {stem}  gradcam {label} ✗ ({e})')

        # ---- composites ----
        def _build_composite(panels, out_name):
            h, w = panels[0][1].shape[:2]
            comp = np.hstack([p[1] for p in panels])
            for i, (lbl, _) in enumerate(panels):
                x = i * w + 6
                cv2.putText(comp, lbl, (x, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(comp, lbl, (x, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            save(Path(args.save_dir) / f'{stem}_{out_name}.jpg', comp)
            print(f'  {stem}  {out_name} ✓')

        # gradcam-only composite (Input + all Grad-CAM targets)
        gc_panels = [(l, o) for l, o in overlays if l.startswith('Grad-CAM')]
        if gc_panels:
            _build_composite([overlays[0]] + gc_panels, 'gradcam_all')

        # full composite (everything)
        if len(overlays) > 2:
            _build_composite(overlays, 'composite')

    print('=' * 60)
    print(f'  Saved → {args.save_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
