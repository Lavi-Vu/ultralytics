#!/usr/bin/env python3
"""Train LightEdge-YOLO on a YOLO-format dataset.

Usage:
    python ultralytics/nn/modules/train_lightedge.py --nano --data coco.yaml --epochs 300 --batch 16
    python ultralytics/nn/modules/train_lightedge.py --small --data coco.yaml --epochs 300 --batch 8 --device 0

Resume from checkpoint:
    python ultralytics/nn/modules/train_lightedge.py --resume /path/to/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

import torch

from ultralytics.nn import LightEdgeYOLO
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Train LightEdge-YOLO")
    g = p.add_argument_group("Model")
    g.add_argument("--nano", action="store_true", help="Nano variant (~3.0M params)")
    g.add_argument("--small", action="store_true", help="Small variant (~8.4M params)")

    g = p.add_argument_group("Data")
    g.add_argument("--data", type=str, default="coco.yaml", help="dataset YAML")
    g.add_argument("--imgsz", type=int, default=640, help="input image size")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=300, help="number of epochs")
    g.add_argument("--batch", type=int, default=16, help="batch size")
    g.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    g.add_argument("--optimizer", type=str, default="AdamW", help="optimizer (SGD, Adam, AdamW)")
    g.add_argument("--weight-decay", type=float, default=0.0005, help="weight decay")
    g.add_argument("--warmup-epochs", type=float, default=3.0, help="warmup epochs")
    g.add_argument("--cos-lr", action="store_true", help="cosine LR schedule")
    g.add_argument("--device", type=str, default="", help="device (e.g. 0, cpu)")
    g.add_argument("--workers", type=int, default=8, help="data loader workers")
    g.add_argument("--project", type=str, default=None, help="project directory")
    g.add_argument("--name", type=str, default=None, help="experiment name")
    g.add_argument("--exist-ok", action="store_true", help="overwrite existing project")
    g.add_argument("--seed", type=int, default=0, help="random seed")
    g.add_argument("--resume", type=str, default=None,
                   help="resume from checkpoint (.pt or directory)")

    g = p.add_argument_group("Validation")
    g.add_argument("--val", action="store_true", default=True,
                   help="run validation after training")
    g.add_argument("--save-period", type=int, default=1, help="save checkpoint every N epochs")

    g = p.add_argument_group("Export")
    g.add_argument("--export", action="store_true", help="export to ONNX after training")
    g.add_argument("--export-format", type=str, default="onnx",
                   choices=["onnx", "torchscript", "tensorrt", "coreml", "tflite"],
                   help="export format")

    g = p.add_argument_group("Augmentation")
    g.add_argument("--mosaic", type=float, default=1.0, help="mosaic augmentation probability")
    g.add_argument("--mixup", type=float, default=0.0, help="mixup augmentation probability")
    g.add_argument("--copy-paste", type=float, default=0.0, help="copy-paste augmentation")
    g.add_argument("--degrees", type=float, default=0.0, help="rotation degrees")
    g.add_argument("--scale", type=float, default=0.5, help="scale factor")
    g.add_argument("--fliplr", type=float, default=0.5, help="horizontal flip probability")

    return p.parse_args()


def resolve_variant(args):
    if args.nano and args.small:
        sys.exit("Specify only one variant: --nano or --small")
    if args.nano:
        return "nano"
    if args.small:
        return "small"
    return "nano"


def main():
    args = parse_args()
    variant = resolve_variant(args)

    project = args.project or "runs/train"
    name = args.name or f"lightedge-{variant}"

    overrides = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "optimizer": args.optimizer,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "device": args.device,
        "workers": args.workers,
        "project": project,
        "name": name,
        "exist_ok": args.exist_ok,
        "seed": args.seed,
        "val": args.val,
        "save_period": args.save_period,
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "copy_paste": args.copy_paste,
        "degrees": args.degrees,
        "scale": args.scale,
        "fliplr": args.fliplr,
        "model": "yolo26n.yaml",
    }

    if args.resume:
        ckpt = Path(args.resume)
        if ckpt.suffix == ".pt":
            model = YOLO(str(ckpt))
            model.train(resume=True)
            return
        overrides["resume"] = True
        trainer = DetectionTrainer(overrides=overrides)
        trainer.train()
        return

    trainer = DetectionTrainer(overrides=overrides)
    nc = trainer.data["nc"] if trainer.data else 80

    print(f"\n{'='*60}")
    print(f"Building LightEdge-YOLO {variant.upper()} ({nc} classes)")
    print(f"{'='*60}\n")

    model = LightEdgeYOLO(variant, nc=nc)
    trainer.model = model
    trainer.model.args = trainer.args
    trainer.model.names = trainer.data.get("names", {})
    trainer.set_model_attributes()
    trainer.module = model

    trainer.train()

    if args.export:
        from ultralytics.nn.modules.lightedge import export_onnx
        export_path = str(Path(trainer.save_dir) / f"lightedge-{variant}.onnx")
        print(f"\nExporting to ONNX: {export_path}")
        export_onnx(trainer.model, export_path, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
