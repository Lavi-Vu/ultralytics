#!/usr/bin/env python3
"""Run controlled LightEdgeDet ablations with one fixed training recipe.

The default study compares the full nano model with no cross-scale fusion, no
reparameterization, and no backbone PSA. Existing runs are never reused, and
incomplete runs cannot contribute metrics to the result table.

Examples:
    python scripts/ablate.py --build-only
    python scripts/ablate.py --data coco.yaml --epochs 100 --batch 8 --device 0 --project runs/ablations/core100
    python scripts/ablate.py --data coco.yaml --epochs 100 --batch 8 --ablate no_sppf backbone_use_sppf=False
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

CORE_ABLATIONS = (
    ("no_cross_fusion", {"neck_use_cross_fusion": False}),
    ("no_reparam", {"backbone_reparam": False, "neck_reparam": False}),
    ("no_psa", {"backbone_attn_stages": [0, 0, 0, 0, 0]}),
)
METRICS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
FIELDS = (
    "name",
    "status",
    "epochs_requested",
    "epochs_logged",
    "batch",
    "seed",
    "params_M",
    "unfused_M",
    "gflops",
    *METRICS,
    "fitness",
    "best_epoch",
    "run_dir",
    "error",
)


def parse_overrides(pairs: list[str]) -> dict:
    """Parse model-key overrides, accepting Python literals for values."""
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"expected key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            overrides[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            overrides[key] = raw
    return overrides


def variants_from_args(args: argparse.Namespace) -> tuple[dict, list[tuple[str, dict]]]:
    """Return baseline and the default core or requested variants."""
    base = yaml.safe_load(Path(args.config).read_text())
    if not isinstance(base, dict):
        raise TypeError("model configuration must be a YAML mapping")
    variants = [("baseline", {})]
    if args.ablate or args.set:
        for group in args.ablate or []:
            variants.append((group[0], parse_overrides(group[1:])))
        for group in args.set or []:
            overrides = parse_overrides(group)
            variants.append(("-".join(overrides), overrides))
    else:
        variants.extend(CORE_ABLATIONS)

    names = set()
    for name, overrides in variants:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in names:
            raise ValueError(f"invalid or duplicate variant name: {name!r}")
        names.add(name)
        if name != "baseline" and not overrides:
            raise ValueError(f"variant {name!r} has no overrides")
        for key, value in overrides.items():
            if key not in base:
                raise ValueError(f"unknown model key {key!r} in {name!r}")
            if base[key] == value:
                raise ValueError(f"{name!r} does not change {key!r}")
    return base, variants


def read_history(path: Path, epochs: int) -> tuple[int, int]:
    """Require one row per requested epoch; return count and best epoch."""
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    observed = [int(row["epoch"]) for row in rows]
    if observed != list(range(1, epochs + 1)):
        raise RuntimeError(f"incomplete training history at {path}: logged {len(rows)}/{epochs} epochs")
    scores = [float(row["metrics/mAP50-95(B)"]) for row in rows]
    if not all(math.isfinite(score) for score in scores):
        raise RuntimeError(f"non-finite validation score in {path}")
    return len(rows), max(range(epochs), key=scores.__getitem__) + 1


def profile_model(config: Path, imgsz: int) -> tuple[float, float, float]:
    """Return unfused parameters, fused parameters, and fused GFLOPs."""
    import torch
    from thop import profile

    from ultralytics.nn.tasks import LightEdgeDetModel

    model = LightEdgeDetModel(str(config), verbose=False).eval()
    unfused = sum(p.numel() for p in model.parameters()) / 1e6
    model.fuse(verbose=False)
    fused = sum(p.numel() for p in model.parameters()) / 1e6
    with torch.no_grad():
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, imgsz, imgsz),), verbose=False)
    return unfused, fused, 2 * macs / 1e9


def run_worker(payload_path: Path) -> None:
    """Train and validate one variant in an isolated process."""
    import zipfile

    from ultralytics import YOLO

    payload = json.loads(payload_path.read_text())
    run_dir = Path(payload["project"]) / payload["name"]
    model = YOLO(payload["config"])
    model.train(
        data=payload["data"],
        epochs=payload["epochs"],
        imgsz=payload["imgsz"],
        batch=payload["batch"],
        workers=payload["workers"],
        device=payload["device"],
        seed=payload["seed"],
        deterministic=True,
        pretrained=False,
        project=payload["project"],
        name=payload["name"],
        exist_ok=False,
        verbose=not payload["quiet"],
    )
    actual_dir = Path(model.trainer.save_dir)
    if actual_dir != run_dir:
        raise RuntimeError(f"unexpected run directory: {actual_dir} (expected {run_dir})")
    epochs_logged, best_epoch = read_history(run_dir / "results.csv", payload["epochs"])
    for checkpoint in (run_dir / "weights/best.pt", run_dir / "weights/last.pt"):
        if not zipfile.is_zipfile(checkpoint):
            raise RuntimeError(f"missing or invalid checkpoint: {checkpoint}")

    metrics = (
        YOLO(str(run_dir / "weights/best.pt"))
        .val(
            data=payload["data"],
            imgsz=payload["imgsz"],
            batch=payload["batch"],
            device=payload["device"],
            project=str(run_dir),
            name="validation",
            exist_ok=False,
            verbose=False,
        )
        .results_dict
    )
    for key in METRICS:
        value = float(metrics[key])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise RuntimeError(f"invalid {key}: {value}")
    result = {"epochs_logged": epochs_logged, "best_epoch": best_epoch, "run_dir": str(run_dir), **metrics}
    (run_dir / "ablation_metrics.json").write_text(json.dumps(result, indent=2) + "\n")


def run_study(args: argparse.Namespace) -> int:
    """Profile and optionally train all variants; write an explicit-status CSV."""
    if args.epochs < 1 or args.imgsz < 64 or args.workers < 0:
        raise ValueError("epochs and imgsz must be positive, and workers cannot be negative")
    if not args.build_only and (not args.data or args.batch is None or args.batch < 1):
        raise ValueError("training requires --data and a fixed positive integer --batch")

    base, variants = variants_from_args(args)
    project = Path(args.project).resolve()
    csv_path = project / "ablation_results.csv"
    if csv_path.exists() or any((project / name).exists() for name, _ in variants):
        raise FileExistsError(f"study already exists in {project}; choose a new --project")
    config_dir = project / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    data = str(Path(args.data).resolve()) if args.data and Path(args.data).exists() else args.data

    failed = False
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        for name, overrides in variants:
            row = dict.fromkeys(FIELDS, "")
            row.update(name=name, status="failed", epochs_requested=args.epochs, batch=args.batch or "", seed=args.seed)
            try:
                config = config_dir / f"{name}.yaml"
                config.write_text(yaml.safe_dump({**base, **overrides}, sort_keys=False))
                row["unfused_M"], row["params_M"], row["gflops"] = profile_model(config, args.imgsz)
                if args.build_only:
                    row["status"] = "profiled"
                else:
                    payload = {
                        "config": str(config),
                        "data": data,
                        "epochs": args.epochs,
                        "imgsz": args.imgsz,
                        "batch": args.batch,
                        "workers": args.workers,
                        "device": args.device,
                        "seed": args.seed,
                        "project": str(project),
                        "name": name,
                        "quiet": args.quiet,
                    }
                    payload_path = config_dir / f"{name}.json"
                    payload_path.write_text(json.dumps(payload, indent=2) + "\n")
                    subprocess.run([sys.executable, __file__, "--worker", str(payload_path)], check=True)
                    row.update(json.loads((project / name / "ablation_metrics.json").read_text()))
                    row["status"] = "complete"
            except Exception as error:  # noqa: BLE001 - record a variant's failure and continue the study
                failed = True
                row["error"] = str(error)
                print(f"{name}: FAILED: {error}", file=sys.stderr, flush=True)
            else:
                print(f"{name}: {row['status']}", flush=True)
            writer.writerow(row)
            file.flush()
    print(f"Results: {csv_path}")
    return 1 if failed else 0


def main() -> int:
    """Parse study options and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ultralytics/cfg/models/lightedgedet_nano.yaml")
    parser.add_argument("--data", help="dataset YAML; required for training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, help="fixed batch for every variant; required for training")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", default="runs/ablations/study")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--ablate", nargs="+", action="append", metavar="NAME_OR_KEY_VALUE")
    parser.add_argument("--set", nargs="+", action="append", metavar="KEY_VALUE")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        return run_study(args)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        run_worker(Path(sys.argv[2]))
    else:
        sys.exit(main())
