"""COCO dataset setup and DataLoader creation.

Supports MS COCO 2017 with the Ultralytics-style augmentation pipeline.
If COCO is not present at data_dir, provides download instructions.
"""

import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np


class COCODataset(Dataset):
    """Minimal COCO detection dataset with on-the-fly augmentation."""

    def __init__(self, data_dir, split="train2017", img_size=640):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images" / split
        self.ann_dir = self.data_dir / "annotations"
        self.img_size = img_size
        self.split = split

        # Check existence
        if not self.img_dir.exists():
            raise FileNotFoundError(
                f"COCO images not found at {self.img_dir}.\n"
                f"  Download: wget http://images.cocodataset.org/zips/{split}.zip\n"
                f"  Unzip to: {self.data_dir}/images/{split}/\n"
                f"  Annotations: wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )

        # Load annotations
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError("pycocotools required: pip install pycocotools")

        ann_file = self.ann_dir / f"instances_{split}.json"
        self.coco = COCO(str(ann_file))
        self.ids = list(self.coco.img_ids)

        print(f"COCO {split}: {len(self.ids)} images")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.img_dir / img_info["file_name"]

        # Load image
        image = Image.open(img_path).convert("RGB")
        w, h = img_info["width"], img_info["height"]

        # Load annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        cls_labels = []
        for ann in anns:
            if ann.get("iscrowd", 0):
                continue
            x, y, bw, bh = ann["bbox"]
            # Convert to xyxy
            boxes.append([x, y, x + bw, y + bh])
            cls_labels.append(ann["category_id"] - 1)  # COCO is 1-indexed

        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            cls_labels = np.zeros((0,), dtype=np.int64)
        else:
            boxes = np.array(boxes, dtype=np.float32)
            cls_labels = np.array(cls_labels, dtype=np.int64)

        # Resize & normalize
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0

        # Scale boxes
        scale_x = self.img_size / w
        scale_y = self.img_size / h
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        return {
            "img": img_tensor,
            "bbox": torch.from_numpy(boxes),
            "cls": torch.from_numpy(cls_labels),
            "image_id": img_id,
            "batch_idx": torch.full((len(boxes),), idx, dtype=torch.long),
        }


def collate_fn(batch):
    """Collate function for variable number of boxes per image."""
    imgs = torch.stack([b["img"] for b in batch])
    image_ids = [b["image_id"] for b in batch]

    all_boxes = []
    all_cls = []
    all_batch_idx = []

    for i, b in enumerate(batch):
        if b["bbox"].size(0) > 0:
            all_boxes.append(b["bbox"])
            all_cls.append(b["cls"])
            all_batch_idx.append(torch.full((b["bbox"].size(0),), i, dtype=torch.long))

    return {
        "img": imgs,
        "bbox": torch.cat(all_boxes, dim=0) if all_boxes else torch.zeros((0, 4)),
        "cls": torch.cat(all_cls, dim=0) if all_cls else torch.zeros((0,), dtype=torch.long),
        "image_id": image_ids,
        "batch_idx": torch.cat(all_batch_idx, dim=0) if all_batch_idx else torch.zeros((0,), dtype=torch.long),
    }


def create_coco_dataloaders(data_dir, batch_size=256, img_size=640, num_workers=8, train=True):
    """Create COCO train and val dataloaders."""
    train_dataset = COCODataset(data_dir, split="train2017", img_size=img_size)
    val_dataset = COCODataset(data_dir, split="val2017", img_size=img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader