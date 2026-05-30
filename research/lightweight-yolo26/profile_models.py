#!/usr/bin/env python3
"""Diagnostic profiling script for all 7 model variants.

Computes exact GFLOPs, param count, and estimated FPS for every model.
Run BEFORE training to verify FLOP targets are met.

Usage:
    python profile.py
    python profile.py --models baseline,neck-ghost --device cpu
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MODEL_REGISTRY as ALL_MODELS


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, default="all",
                        help="Comma-separated model names or 'all'")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--fps-trials", type=int, default=100,
                        help="Number of warmup+trials for FPS benchmark")
    return parser.parse_args()


def profile_model(model_cls, name, device, img_size, fps_trials=100):
    print(f"\n{'='*60}")
    print(f"  Model: {name}")
    print(f"{'='*60}")

    try:
        from thop import profile as thop_profile
    except ImportError:
        thop_profile = None

    # Build
    model = model_cls(nc=80, reg_max=1, end2end=True)
    model = model.to(device)
    model.eval()

    input_tensor = torch.randn(1, 3, img_size, img_size).to(device)

    # ---- Params ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # ---- Fuse reparam models for correct inference FLOPs ----
    if hasattr(model, "fuse"):
        model.fuse()
        model.eval()

    # ---- GFLOPs via thop ----
    if thop_profile is not None:
        flops, _ = thop_profile(model, inputs=(input_tensor,), verbose=False)
        gflops = flops / 1e9
    else:
        gflops = 0.0
        print("  [WARN] thop not installed — install via: pip install thop")

    # ---- FPS benchmark ----
    # Warmup
    for _ in range(50):
        _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()

    # Timed
    start = time.time()
    for _ in range(fps_trials):
        _ = model(input_tensor)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = fps_trials / elapsed

    # ---- Model-specific notes ----
    notes = ""
    if hasattr(model, "gate_frac_active"):
        notes = f" | Gate active: {model.gate_frac_active():.2%}"

    print(f"  Params:        {total_params/1e6:.2f}M ({total_params:,})")
    print(f"  Trainable:     {trainable_params/1e6:.2f}M")
    print(f"  GFLOPs:        {gflops:.2f} (at {img_size}x{img_size})")
    print(f"  FPS:           {fps:.1f} (batch=1, {device}){notes}")
    print(f"  Latency:       {elapsed/fps_trials*1000:.2f} ms")

    return {
        "model": name,
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

    if args.models == "all":
        model_names = list(ALL_MODELS.keys())
    else:
        model_names = [m.strip() for m in args.models.split(",")]

    results = []
    for name in model_names:
        if name not in ALL_MODELS:
            print(f"Unknown model: {name}. Available: {list(ALL_MODELS.keys())}")
            continue
        r = profile_model(ALL_MODELS[name], name, device, args.img_size, args.fps_trials)
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

    # Save to JSON
    import json
    out_path = Path("profiling_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()