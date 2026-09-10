#!/usr/bin/env python3
"""Ablation harness for LightEdgeDet (ultralytics fork).

Builds a baseline model from a YAML, applies named YAML-key overrides for each ablation
variant, runs build + (optionally) train -> val + fused profile, and prints a markdown
comparison table.

Usage
-----
# Build-only (no GPU/dataset): prints fused params/GFLOPs for every variant
python scripts/ablate.py --build-only

# Full runs (needs GPU + dataset):
python scripts/ablate.py --data coco8.yaml --epochs 50 --device 0 \
    --ablate reparam_off backbone_reparam=False \
    --ablate no_cross_fusion neck_use_cross_fusion=False \
    --ablate k3 backbone_kernel_sizes=[3,3,3,3,3] \
    --ablate head_shared head_type=shared head_channels=64

# Short form: --set auto-names the variant from the first overridden key
python scripts/ablate.py --data coco.yaml --epochs 300 --device 0 \
    --set neck_use_cross_fusion=False \
    --set backbone_psa_blocks=2

Run from the ultralytics fork root so relative cfg paths resolve.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gc
import sys
from pathlib import Path

import yaml
import torch

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_value(raw: str):
    """Parse an override value: literals (list/bool/int/float) fall back to str."""
    raw = raw.strip()
    try:
        return ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return raw


def parse_overrides(pairs):
    """Split ['k=v', ...] into {k: v}."""
    out = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override must be key=value, got '{p}'")
        k, v = p.split("=", 1)
        out[k.strip()] = parse_value(v)
    return out


def build_arg_parser():
    p = argparse.ArgumentParser(description="LightEdgeDet ablation harness")
    p.add_argument("--config", default="ultralytics/cfg/models/lightedgedet_nano.yaml", help="base model YAML")
    p.add_argument("--data", default=None, help="dataset YAML (required for training)")
    p.add_argument("--epochs", type=int, default=50, help="training epochs per variant")
    p.add_argument("--imgsz", type=int, default=640, help="image size")
    p.add_argument("--batch", type=int, default=None, help="batch size (default: from YAML)")
    p.add_argument("--workers", type=int, default=None, help="dataloader workers (default: auto)")
    p.add_argument("--device", default="", help="device, e.g. 0 or cpu")
    p.add_argument("--project", default="runs/ablations", help="output dir for runs and configs")
    p.add_argument("--build-only", action="store_true", help="only build + profile each variant, no training")
    p.add_argument(
        "--baseline-weights",
        default=None,
        help="existing trained baseline best.pt; when set, the 'full'/'baseline' "
        "variant is profiled and validated against these weights WITHOUT retraining",
    )
    p.add_argument(
        "--ablate",
        nargs="+",
        action="append",
        metavar=("NAME", "key=value"),
        help="ablation variant: a name followed by key=value overrides (repeatable)",
    )
    p.add_argument(
        "--set",
        nargs="+",
        action="append",
        metavar="key=value",
        help="quick ablation: overrides, variant auto-named from keys (repeatable)",
    )
    p.add_argument("--quiet", action="store_true", help="reduce trainer output")
    return p


# ---------------------------------------------------------------------------
# Variant construction
# ---------------------------------------------------------------------------


def build_variants(args):
    base_path = Path(args.config)
    base = yaml.safe_load(base_path.read_text())
    variants = [("baseline", {})]

    for group in args.ablate or []:
        name = group[0]
        variants.append((name, parse_overrides(group[1:])))
    for group in args.set or []:
        overrides = parse_overrides(group)
        name = "-".join(overrides.keys())
        variants.append((name, overrides))

    # de-duplicate by name, keep first occurrence
    seen, out = set(), []
    for name, ov in variants:
        if name in seen:
            print(f"! duplicate variant name '{name}' skipped")
            continue
        seen.add(name)
        out.append((name, ov))
    return base, out


def write_variant_yaml(base, overrides, cfg_dir: Path, name: str) -> Path:
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(base)
    cfg.update(overrides)
    path = cfg_dir / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


def profile_model(yaml_path: str | Path, imgsz: int):
    """Build + fuse + profile a variant.

    Returns (unfused_M, fused_M, gflops). unfused_M is the training-time param count
    (reparam branches present when backbone_reparam=True), fused_M is the deployment
    count after fuse().
    """
    import torch
    from thop import profile

    from ultralytics.nn.tasks import LightEdgeDetModel

    m = LightEdgeDetModel(str(yaml_path), verbose=False)
    unfused = sum(p.numel() for p in m.parameters()) / 1e6
    m.fuse()
    m.eval()
    fused = sum(p.numel() for p in m.parameters()) / 1e6
    with torch.no_grad():
        macs, _ = profile(m, inputs=(torch.zeros(1, 3, imgsz, imgsz),), verbose=False)
    gflops = macs * 2 / 1e9
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return unfused, fused, gflops


# ---------------------------------------------------------------------------
# Train + validate a variant
# ---------------------------------------------------------------------------


def train_and_val(args, name: str, yaml_path: Path, weights: str | None = None):
    """Train a variant (or reuse existing weights) and return its metrics dict.

    If ``weights`` is given, training is skipped and the variant is validated
    directly against that checkpoint (used for the already-trained baseline).
    """
    import torch
    from ultralytics import YOLO

    if weights is not None:
        print(f"    [reuse] skipping training, validating {weights}", flush=True)
        best_pt = weights
    else:
        train_kwargs = {
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "device": args.device,
            "project": args.project,
            "name": name,
            "exist_ok": True,
            "verbose": not args.quiet,
        }
        if args.batch is not None:
            train_kwargs["batch"] = args.batch
        if args.workers is not None:
            train_kwargs["workers"] = args.workers

        train_model = YOLO(str(yaml_path))
        train_model.train(**train_kwargs)
        del train_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Resolve the actual save dir the same way the trainer does (anchored on
        # SETTINGS['runs_dir']): runs_dir / task / project / name.
        from types import SimpleNamespace

        from ultralytics.engine.trainer import get_save_dir

        save_dir = get_save_dir(
            SimpleNamespace(task="lightedgedet", project=args.project, name=name, mode="train", exist_ok=True)
        )
        best_pt = save_dir / "weights" / "best.pt"
        if not best_pt.exists():
            candidates = sorted(Path(args.project).glob(f"**/{name}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
            if not candidates:
                raise FileNotFoundError(f"best.pt not found under {save_dir}")
            best_pt = candidates[-1]

    model = YOLO(str(best_pt))
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch or 16,
        verbose=False,
    )
    results = metrics.results_dict
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

METRIC_KEYS = {
    "mAP50-95": "metrics/mAP50-95(B)",
    "mAP50": "metrics/mAP50(B)",
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
}


def main():
    args = build_arg_parser().parse_args()
    if not args.build_only and args.data is None:
        sys.exit("error: --data is required unless --build-only is given")
    if args.build_only and args.data is not None:
        print("! --build-only given: training is skipped regardless of --data")

    base, variants = build_variants(args)
    cfg_dir = Path(args.project) / "configs"
    print(f"base config : {args.config}")
    print(f"variants    : {[n for n, _ in variants]}")
    print(f"build only  : {args.build_only}")
    print()

    rows = []
    for name, overrides in variants:
        ov_desc = " ".join(f"{k}={v}" for k, v in overrides.items()) or "-"
        print(f"[{name}] overrides: {ov_desc}", flush=True)
        try:
            yaml_path = write_variant_yaml(base, overrides, cfg_dir, name)
            unfused, fused, gflops = profile_model(yaml_path, args.imgsz)
            print(f"  profile: {fused:.3f}M params fused ({unfused:.3f}M unfused), {gflops:.2f} GFLOPs", flush=True)

            # the unchanged full/baseline variant reuses existing weights if provided
            reuse = None
            if args.baseline_weights and name in ("full", "baseline") and not overrides:
                reuse = args.baseline_weights

            metrics = {} if args.build_only else train_and_val(args, name, yaml_path, reuse)
            rows.append({"name": name, "params_M": fused, "unfused_M": unfused, "gflops": gflops, **metrics})
        except Exception as e:  # noqa: BLE001 - isolate failures so one variant never aborts the batch
            print(f"  FAILED: {e}", flush=True)
            rows.append({"name": name, "params_M": float("nan"), "unfused_M": float("nan"), "gflops": float("nan")})
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- report ----
    out = ["| variant | params(M) | unfused(M) | GFLOPs |"]
    out += ["|---|---|---|---|"]
    if not args.build_only:
        for disp, key in METRIC_KEYS.items():
            out[0] += f" {disp} |"
            out[1] += "---|"
    out[0] += ""
    out[1] += ""
    for r in rows:
        pm = "—" if r["params_M"] != r["params_M"] else f"{r['params_M']:.3f}"
        um = "—" if r["unfused_M"] != r["unfused_M"] else f"{r['unfused_M']:.3f}"
        gf = "—" if r["gflops"] != r["gflops"] else f"{r['gflops']:.2f}"
        line = f"| {r['name']} | {pm} | {um} | {gf} |"
        if not args.build_only:
            for key in METRIC_KEYS.values():
                v = r.get(key)
                line += f" {v if v is None else f'{v:.4f}'} |"
        out.append(line)
    print()
    print("\n".join(out))

    csv_path = Path(args.project) / "ablation_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nresults written to {csv_path}")


if __name__ == "__main__":
    main()
