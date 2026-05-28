#!/usr/bin/env python3
"""Training entrypoint for lightweight YOLO26 research experiments.

Usage:
    python train.py --model baseline --batch 256 --epochs 300
    python train.py --model reparam --batch 128 --epochs 300
    python train.py --model neck-ghost --batch 256 --epochs 300
    python train.py --model gate --batch 256 --epochs 300 --gate-lambda 1e-4
"""

import argparse
import os
import sys
import time
import math
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # ultralytics root
sys.path.insert(0, str(Path(__file__).resolve().parent))  # research/lightweight-yolo26

# Import our model zoo
from models import (
    YOLO26NBaseline, YOLO26NReparam, YOLO26NGate,
    YOLO26NCombined, YOLO26NeckGhost, YOLO26NeckLight, YOLO26NeckWide
)
from dataset_setup import create_coco_dataloaders

import yaml


MODEL_REGISTRY = {
    "baseline": YOLO26NBaseline,
    "reparam": YOLO26NReparam,
    "gate": YOLO26NGate,
    "combined": YOLO26NCombined,
    "neck-ghost": YOLO26NeckGhost,
    "neck-light": YOLO26NeckLight,
    "neck-wide": YOLO26NeckWide,
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="baseline", choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--data-dir", type=str, default="./coco")
    parser.add_argument("--output-dir", type=str, default="./runs")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--gate-lambda", type=float, default=1e-4, help="L1 gate sparsity weight")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, scheduler, scaler, epoch, args, writer):
    """Run one training epoch."""
    model.train()
    device = next(model.parameters()).device
    num_batches = len(loader)

    epoch_loss = 0.0
    epoch_cls = 0.0
    epoch_box = 0.0
    epoch_dfl = 0.0

    for batch_idx, batch in enumerate(loader):
        images = batch["img"].to(device)
        cls_labels = batch["cls"].to(device)
        box_labels = batch["bbox"].to(device)
        batch_idx_tensor = batch["batch_idx"].to(device)

        # Warmup LR
        if epoch < args.warmup_epochs:
            warmup_iters = args.warmup_epochs * num_batches
            current_iter = epoch * num_batches + batch_idx
            lr_scale = min(1.0, (current_iter + 1) / warmup_iters)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * args.lr

        with autocast():
            outputs = model(images)
            loss_dict = compute_loss(outputs, cls_labels, box_labels, batch_idx_tensor, model)
            loss = loss_dict["loss"]

            # Channel gate sparsity regularization (for gated models)
            if hasattr(model, "gate_sparsity_loss"):
                gate_loss = model.gate_sparsity_loss()
                loss = loss + args.gate_lambda * gate_loss
                loss_dict["gate"] = gate_loss.item()

        # Backward
        scaler.scale(loss).backward()

        if (batch_idx + 1) % args.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += loss_dict["loss"]
        epoch_cls += loss_dict.get("cls", 0.0)
        epoch_box += loss_dict.get("box", 0.0)
        epoch_dfl += loss_dict.get("dfl", 0.0)

        # Logging
        if batch_idx % 50 == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:3d} | Batch {batch_idx:4d}/{num_batches} | "
                  f"Loss {loss_dict['loss']:.4f} | LR {lr_now:.2e}")

    # End of epoch
    avg_loss = epoch_loss / num_batches
    print(f"--- Epoch {epoch:3d} complete | Avg Loss: {avg_loss:.4f} "
          f"| Cls: {epoch_cls/num_batches:.4f} | Box: {epoch_box/num_batches:.4f}")

    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
    if hasattr(model, "gate_frac_active"):
        frac = model.gate_frac_active()
        writer.add_scalar("train/gate_frac_active", frac, epoch)

    scheduler.step()


def compute_loss(outputs, cls_labels, box_labels, batch_idx_tensor, model):
    """Simplified E2E-style loss computation.

    In production, this would use Ultralytics' E2ELoss. For research,
    we use a simplified TAL-like loss matching the YOLO26 paradigm.
    """
    cls_loss = torch.tensor(0.0, device=model.device)
    box_loss = torch.tensor(0.0, device=model.device)

    if isinstance(outputs, dict):
        if "one2many" in outputs:
            pred = outputs["one2many"]
        else:
            pred = outputs
    else:
        pred = {"boxes": outputs, "scores": outputs}

    boxes = pred.get("boxes", torch.zeros(1))
    scores = pred.get("scores", torch.zeros(1))
    # Simplified BCE + CIoU loss (placeholder — real E2ELoss is deep-copied from Ultralytics)

    total = cls_loss + box_loss
    return {"loss": total, "cls": cls_loss.item(), "box": box_loss.item(), "dfl": 0.0}


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output dir
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tensorboard"))

    # Save config
    with open(out_dir / "config.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # Build model
    print(f"Building model: {args.model}")
    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(nc=80, reg_max=1, end2end=True)

    # Profile
    from thop import profile
    input_test = torch.randn(1, 3, args.img_size, args.img_size)
    flops, params = profile(model, inputs=(input_test,), verbose=False)
    gflops = flops / 1e9
    params_m = params / 1e6
    print(f"  Params: {params_m:.2f}M | GFLOPs: {gflops:.2f}")
    writer.add_scalar("model/params_m", params_m, 0)
    writer.add_scalar("model/gflops", gflops, 0)

    model = model.to(device)
    model.device = device

    # Data
    print(f"Loading COCO from {args.data_dir} ...")
    train_loader, val_loader = create_coco_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch,
        img_size=args.img_size,
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.warmup_epochs
    )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Resume
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    best_map = 0.0

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, train_loader, optimizer, scheduler, scaler, epoch, args, writer)

        # Validation every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            map_metrics = validate(model, val_loader, device, epoch, writer)
            if map_metrics.get("mAP", 0) > best_map:
                best_map = map_metrics.get("mAP", 0)
                ckpt_path = out_dir / "best.pt"
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "mAP": best_map,
                    "args": vars(args),
                }, ckpt_path)
                print(f"  New best model saved: mAP={best_map:.3f}")

        # Save checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt_path = out_dir / f"epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, ckpt_path)

    print(f"Training complete. Best mAP: {best_map:.3f}")
    print(f"Model and logs saved to {out_dir}")
    writer.close()


def validate(model, loader, device, epoch, writer):
    """Run COCO validation and log metrics."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["img"].to(device)

            with autocast():
                outputs = model(images)

            # Extract predictions (simplified — full COCO eval in val.py)
            if isinstance(outputs, dict):
                if "one2one" in outputs:
                    pred = outputs["one2one"]
                elif "one2many" in outputs:
                    pred = outputs["one2many"]
                else:
                    pred = outputs
            else:
                pred = outputs

            if batch_idx % 20 == 0:
                print(f"  Val batch {batch_idx}/{len(loader)}")

    # For now return placeholder metrics
    metrics = {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    if writer:
        for k, v in metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
    return metrics


if __name__ == "__main__":
    main()