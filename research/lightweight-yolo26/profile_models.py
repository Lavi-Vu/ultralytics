#!/usr/bin/env python3
"""Profile all 7 model variants by building from YAML configs.

Builds each model the same way train_ultralytics.py does (via ultralytics_patch),
then measures params, GFLOPs, and FPS.

Usage:
    python profile_models.py
    python profile_models.py --models yolo26n_baseline --device cpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))

import ultralytics_patch  # noqa: F401

from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated YAML names or 'all'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--fps-trials", type=int, default=100,
                        help="Number of warmup+trials for FPS benchmark")
    return parser.parse_args()


# Map short names used by train.sh to YAML filenames
YAML_MAP = {
    "baseline": "yolo26n_baseline",
    "reparam": "yolo26n_reparam",
    "gate": "yolo26n_gate",
    "combined": "yolo26n_combined",
    "neck-ghost": "yolo26n_neck_ghost",
    "neck-light": "yolo26n_neck_light",
    "neck-wide": "yolo26n_neck_wide",
}


def profile_yaml(yaml_stem, device, img_size, fps_trials=100):
    """Build model from YAML config and profile it."""
    yaml_path = f"cfg/{yaml_stem}.yaml"
    display_name = yaml_stem.replace("yolo26n_", "")

    print(f"\n{'='*60}")
    print(f"  Model: {display_name}  ({yaml_path})")
    print(f"{'='*60}")

    # Build model via YOLO to get DetectionModel
    yolo = YOLO(yaml_path)
    model = yolo.model
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, img_size, img_size).to(device)

    # Params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # GFLOPs via thop
    try:
        from thop import profile as thop_profile
        flops, _ = thop_profile(model, inputs=(input_tensor,), verbose=False)
        gflops = flops / 1e9
    except ImportError:
        gflops = 0.0
        print("  [WARN] thop not installed — install via: pip install thop")

    # FPS benchmark
    for _ in range(50):
        _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(fps_trials):
        _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = fps_trials / elapsed

    print(f"  Params:        {total_params/1e6:.2f}M ({total_params:,})")
    print(f"  Trainable:     {trainable_params/1e6:.2f}M")
    print(f"  GFLOPs:        {gflops:.2f} (at {img_size}x{img_size})")
    print(f"  FPS:           {fps:.1f} (batch=1, {device})")
    print(f"  Latency:       {elapsed/fps_trials*1000:.2f} ms")

    return {
        "model": display_name,
        "yaml": yaml_stem,
        "params_m": round(total_params / 1e6, 4),
        "trainable_m": round(trainable_params / 1e6, 4),
        "gflops": round(gflops, 2),
        "fps": round(fps, 1),
        "latency_ms": round(elapsed / fps_trials * 1000, 2),
    }


def main():
    args = get_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    cfg_dir = Path(__file__).resolve().parent / "cfg"

    if args.models == "all":
        yaml_stems = list(YAML_MAP.values())
    else:
        names = [m.strip() for m in args.models.split(",")]
        yaml_stems = [YAML_MAP.get(n, n) for n in names]

    results = []
    for stem in yaml_stems:
        yaml_file = cfg_dir / f"{stem}.yaml"
        if not yaml_file.exists():
            print(f"  [SKIP] {stem}.yaml not found in cfg/")
            continue
        r = profile_yaml(stem, device, args.img_size, args.fps_trials)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Model':<18} {'Params(M)':<12} {'GFLOPs':<10} {'FPS':<10} {'Lat(ms)':<10}")
    print(f"  {'-'*60}")
    for r in results:
        print(f"  {r['model']:<18} {r['params_m']:<12.2f} {r['gflops']:<10.2f} "
              f"{r['fps']:<10.1f} {r['latency_ms']:<10.2f}")

    out_path = Path("profiling_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()