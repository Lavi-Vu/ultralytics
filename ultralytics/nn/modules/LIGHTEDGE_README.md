# LightEdge-YOLO

**Lightweight NMS-free object detection for edge deployment.**

LightEdge-YOLO is a custom architecture built on the Ultralytics framework, designed to compete with YOLO26-Nano/Small in accuracy while enabling efficient deployment on resource-constrained devices (Raspberry Pi, Jetson Nano, mobile phones).

## Key Design Goals

| Goal | Approach |
|------|----------|
| **Lightweight** | GhostConv + depthwise separable design keeps params low |
| **NMS-free** | One-to-one decoupled head removes NMS post-processing |
| **Small object accuracy** | Preserved fine-grained P2 features + SmallObjectBoostLoss |
| **Edge-friendly** | Quantization-stable activations, reparameterizable convs |
| **Fast inference** | No DFL, direct box regression, fusable branches |

## Variants

| Variant | Params | GFLOPs | Target Device |
|---------|--------|--------|---------------|
| **Nano** | 3.0M | 13.5 | Raspberry Pi 4/5, Jetson Nano, mobile CPU |
| **Small** | 8.4M | 34.3 | Jetson Orin, phone GPU, edge servers |

## Quick Start

```python
from ultralytics.nn import LightEdgeYOLO

# Create model
model = LightEdgeYOLO("nano", nc=80)

# Inference
model.eval()
import torch
x = torch.randn(1, 3, 640, 640)
with torch.no_grad():
    out = model(x)
# out is (preds_tensor, preds_dict) in eval mode
# preds_tensor shape: (1, 300, 6) = [x1,y1,x2,y2,score,class]

# Training mode returns dict for loss computation
model.train()
out = model(x)
# out = {"one2many": {...}, "one2one": {...}}
```

## Architecture Overview

```
Input (3×H×W)
  │
  ├── Stem Conv3×3, stride=2
  │
  ├── Stage 1 (stride 4) ─── P2 ───┐
  │   └── 2× RepViTGhostBlock       │
  │                                  │
  ├── Stage 2 (stride 8) ─── P3 ────┤
  │   └── 3× RepViTGhostBlock        │
  │                                  │
  ├── Stage 3 (stride 16) ── P4 ────┤──► Adaptive BiFPN ──► LightEdge Head
  │   └── 4× RepViTGhostBlock        │       │                   │
  │                                  │    weighted fusion    ┌───┴───┐
  ├── Stage 4 (stride 32) ── P5 ────┘    top-down +         cls    reg
  │   └── 2× RepViTGhostBlock             bottom-up          (80)   (4)
  │
  └── Output: 300 detections (NMS-free top-k)
```

### Core Building Block: `RepViTGhostBlock`

```
Input
  ├── GhostConv (cheap expansion, ~50% fewer params than standard conv)
  ├── RepDWConv 3×3 (multi-branch DW, fuses to single conv at inference)
  ├── Coordinate Attention (spatial attention via 1D H/W pooling)
  ├── ECA (efficient channel attention, 1D conv over channels)
  ├── Projection Conv 1×1
  └── + Skip connection (when channels match)
```

## Training

Training uses Varifocal Loss for classification + GIoU for box regression, with SmallObjectBoostLoss weighting small objects higher to improve AP_S.

### Using Ultralytics Trainer

The trainer expects a model string path. Override the model after creation:

```python
from ultralytics.nn import LightEdgeYOLO
from ultralytics.models.yolo.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={
    "model": "yolo26n.yaml",  # dummy path, replaced below
    "data": "coco.yaml",
    "epochs": 300,
    "batch": 16,
    "imgsz": 640,
    "device": 0,
})
model = LightEdgeYOLO("nano", nc=trainer.data["nc"])
trainer.model = model
trainer.model.args = trainer.args
trainer.model.names = trainer.data["names"]
trainer.set_model_attributes()
trainer.module = model

trainer.train()
```

### Custom Training Loop

The model handles loss internally — pass the batch dict to `model()` in training mode:

```python
from ultralytics.nn import LightEdgeYOLO
import torch

model = LightEdgeYOLO("nano", nc=80).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for batch in dataloader:
    loss, loss_items = model(batch)      # returns (loss, [box_loss, cls_loss])
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## Export

The model supports ONNX, TensorRT, CoreML, and TFLite export:

```python
from ultralytics.nn.modules.lightedge import LightEdgeYOLO, export_onnx

model = LightEdgeYOLO("nano")
model.eval()

# ONNX
export_onnx(model, "lightedge_nano.onnx")

# Export mode returns one-shot top-k detections
model.head.export = True
model.head.format = "onnx"
x = torch.randn(1, 3, 640, 640)
out = model(x)  # (1, 300, 6) — no tuple wrapping
```

Supported formats:

| Format | Function | Notes |
|--------|----------|-------|
| ONNX | `export_onnx()` | opset 17, supports dynamic batching |
| TensorRT | `export_tensorrt()` | FP16 by default, via ONNX intermediate |
| CoreML | `export_coreml()` | MLProgram format, ImageType input |
| TFLite | `export_tflite()` | Optional INT8 quantization |

## Model Fusion

Fuse Conv+BN and reparameterize depthwise convolutions for inference:

```python
model.fuse()  # fuses all eligible layers
x = torch.randn(1, 3, 640, 640)
out = model(x)  # inference with fused graph
```

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **No DFL** | Direct 4-coord regression is simpler, faster, and more quantization-friendly |
| **GhostConv over standard Conv** | ~50% fewer FLOPs for same output channels |
| **RepDWConv** | Multi-branch training improves accuracy; fuses to single path at inference |
| **CoordAtt + ECA** | Adds ~0 params while significantly improving small-object localization |
| **Weighted BiFPN** | Learnable fusion weights adapt to different scales, improving feature mixing |
| **NMS-free head** | Removes NMS latency and simplifies deployment pipeline |
| **One-to-one + one2many** | Dual assignment gives both clean inference and rich training signal |

## File Structure

```
ultralytics/nn/modules/
├── lightedge.py               # Complete implementation
├── LIGHTEDGE_README.md        # This file
├── LIGHTEDGE_ARCHITECTURE.md  # Detailed architecture documentation
```

## Requirements

- PyTorch >= 1.8
- Ultralytics (for framework integration)
- For export: onnx, onnxslim, coremltools, tensorflow (optional)

## License

Same as Ultralytics — AGPL-3.0
