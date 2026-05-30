#!/usr/bin/env python3
"""Train research model variants using the standard Ultralytics pipeline.

All YAML configs describe their architecture directly with custom module names
(ReparamC3k2, GatedC3k2, GhostC3k2). Importing ultralytics_patch auto-registers
these names into parse_model's frozensets so width/depth scaling works natively.

Usage:
    python train_ultralytics.py --cfg cfg/yolo26n_reparam.yaml --data coco.yaml
    python train_ultralytics.py --cfg cfg/yolo26n_gate.yaml
"""

import argparse
import sys
from pathlib import Path

# Apply monkey-patch BEFORE any YOLO import
_research_dir = str(Path(__file__).resolve().parent)
if _research_dir not in sys.path:
    sys.path.append(_research_dir)  # append, not insert(0) — avoids shadowing stdlib

import ultralytics_patch  # noqa: F401 — auto-registers custom modules in parse_model

from ultralytics import YOLO


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, required=True)
    p.add_argument("--data", type=str, default="coco.yaml")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--project", type=str, default="runs/research")
    p.add_argument("--name", type=str, default=None)
    return p.parse_args()


def main():
    args = get_args()
    print(f"Loading model from {args.cfg}...")
    yolo = YOLO(args.cfg)
    yolo.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        project=args.project,
        name=args.name or Path(args.cfg).stem,
        device=args.device,
    )


if __name__ == "__main__":
    main()