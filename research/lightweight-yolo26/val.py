#!/usr/bin/env python3
"""COCO validation script for lightweight YOLO26 research models.

Computes full COCO mAP@0.5:0.95 using pycocotools.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import MODEL_REGISTRY
from dataset_setup import create_coco_dataloaders


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="./coco")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.001)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = MODEL_REGISTRY[args.model](nc=80, reg_max=1, end2end=True)
    ckpt = torch.load(args.weights, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model = model.to(device).eval()
    print(f"Loaded {args.model} from {args.weights}")

    # Data
    _, val_loader = create_coco_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch,
        img_size=args.img_size,
        train=False,
    )

    # Profile
    from thop import profile
    input_test = torch.randn(1, 3, args.img_size, args.img_size).to(device)
    flops, params = profile(model, inputs=(input_test,), verbose=False)
    print(f"Params: {params/1e6:.2f}M | GFLOPs: {flops/1e9:.2f}")

    # Run inference
    results = []
    image_id = 0

    for batch in val_loader:
        images = batch["img"].to(device)
        img_ids = batch.get("image_id", torch.arange(images.size(0)) + image_id)
        image_id += images.size(0)

        outputs = model(images)
        if isinstance(outputs, dict):
            if "one2one" in outputs:
                pred = outputs["one2one"]
                boxes = pred.get("boxes", torch.zeros(1))
                scores = pred.get("scores", torch.zeros(1))
            else:
                continue
        else:
            continue  # raw tensor, skip for now

        for b in range(images.size(0)):
            dets = boxes[b]  # (N, 4) or (N, K)
            scrs = scores[b]  # (N, nc)
            if scrs.ndim == 2:
                cls_scores, cls_ids = scrs.max(dim=-1)
            else:
                cls_scores = scrs
                cls_ids = torch.zeros_like(scrs, dtype=torch.long)

            mask = cls_scores > args.conf_thres
            dets = dets[mask]
            cls_scores = cls_scores[mask]
            cls_ids = cls_ids[mask]

            for i in range(dets.shape[0]):
                x1, y1, x2, y2 = dets[i].tolist()
                w, h = x2 - x1, y2 - y1
                results.append({
                    "image_id": int(img_ids[b]),
                    "category_id": int(cls_ids[i]) + 1,
                    "bbox": [x1, y1, w, h],
                    "score": float(cls_scores[i]),
                })

    # COCO eval
    ann_file = Path(args.data_dir) / "annotations" / "instances_val2017.json"
    coco_gt = COCO(str(ann_file))
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = {
        "mAP@0.5:0.95": coco_eval.stats[0],
        "mAP@0.5": coco_eval.stats[1],
        "mAP@0.75": coco_eval.stats[2],
    }
    print(f"\nResults: {stats}")
    return stats


if __name__ == "__main__":
    main()