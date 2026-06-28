# LightEdge-YOLO Architecture

## Design Philosophy

LightEdge-YOLO is designed from first principles for **edge deployment**. Every component is chosen to balance accuracy, speed, and quantization stability on resource-constrained hardware. The architecture targets three core metrics:

1. **Parameter efficiency** — achieve competitive mAP with <3M (Nano) or <9M (Small) params
2. **Latency predictability** — no NMS, no DFL, no dynamic branches during inference
3. **Quantization readiness** — use SiLU activations, avoid softmax in decode path, prefer additive fusion

---

## High-Level Architecture

```
                     ┌─────────────────────────────────────────────────────┐
                     │                   LightEdge Head                     │
                     │  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
                     │  │  P3/8    │  │  P4/16   │  │  P5/32   │          │
                     │  │cls+reg   │  │cls+reg   │  │cls+reg   │          │
                     │  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
                     │       │              │              │               │
                     │       └──────────┬───┴───┬──────────┘               │
                     │                  │       │                          │
                     │           ┌──────▼───────▼──────┐                   │
                     │           │   Adaptive BiFPN    │                   │
                     │           │  (weighted fusion)  │                   │
                     │           └──────┬───────┬──────┘                   │
                     │    P3/8 ▲        │P4/16  │P5/32                    │
                     │         │        │       │                          │
Backbone:            │    ┌────┴────────┴───────┴───┐                      │
RepViTGhost          │    │  Stage 2   Stage 3  Stage4                    │
Hybrid               │    │  (stride8) (stride16)(stride32)               │
                     │    └─────────────────────────┘                      │
                     │              ▲                                      │
                     │              │                                      │
                     │    ┌─────────┴──────────┐                          │
                     │    │     Stage 1          │                          │
                     │    │   (stride 4) P2     │── Auxiliary supervision  │
                     │    └─────────┬──────────┘   (small objects)         │
                     │              ▲                                      │
                     │              │                                      │
                     │    ┌─────────┴──────────┐                          │
                     │    │   Stem Conv 3×3 s2  │                          │
                     │    └─────────┬──────────┘                          │
                     │              │                                      │
                     │         Input Image                                 │
                     └─────────────────────────────────────────────────────┘
```

---

## 1. Backbone: RepViTGhost Hybrid

### Design

The backbone is a 4-stage hierarchical feature extractor inspired by RepViT and GhostNet. It outputs multi-scale feature maps at strides 4 (P2), 8 (P3), 16 (P4), and 32 (P5).

### Stem

```
Input (3, H, W)
  └── Conv(k=3, s=2) → BN → SiLU
  └── Output: (C_stem, H/2, W/2)
```

### Stages

Each stage begins with a stride-2 Conv for downsampling, followed by stacked `RepViTGhostBlock` modules.

| Stage | Stride | Nano Channels | Nano Depth | Small Channels | Small Depth |
|-------|--------|---------------|------------|----------------|-------------|
| 1     | 4      | 32            | 2          | 48             | 3           |
| 2     | 8      | 64            | 3          | 96             | 4           |
| 3     | 16     | 128           | 4          | 192            | 6           |
| 4     | 32     | 256           | 2          | 384            | 3           |

### RepViTGhostBlock

```
                    Input (C_in)
                       │
                       ▼
              ┌────────────────┐
              │   GhostConv    │  ← Cheap expansion: primary 1×1 + cheap 3×3 DW
              │  C_in → C_mid  │     C_mid = C_in × expand_ratio
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   RepDWConv    │  ← Multi-branch depthwise (3×3 + 1×1 + identity)
              │  C_mid → C_mid │     Fuses to single 3×3 DW during inference
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │   CoordAtt     │  ← 1D H-pool + 1D W-pool → conv → sigmoid
              │  C_mid → C_mid │     Captures long-range spatial dependencies
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │     ECA        │  ← GAP → Conv1d(k≈5) → sigmoid
              │  C_mid → C_mid │     Local cross-channel interaction (≈0 params)
              └────────┬───────┘
                       │
                       ▼
              ┌────────────────┐
              │ Project Conv1×1│  ← C_mid → C_out
              └────────┬───────┘
                       │
              ┌────────┴────────┐
              │  + identity if  │
              │  C_in == C_out  │
              └────────┬────────┘
                       │
                       ▼
                    SiLU activation
                       │
                       ▼
                    Output (C_out)
```

**GhostConv** halves parameter cost vs standard Conv by splitting channels into a primary 1×1 path and a cheap 3×3 depthwise path:
```python
# Standard expansion: params = C_in × C_mid × k²
# Ghost expansion:   params = C_in × C_out/2 × 1² + C_out/2 × 1 × 3² × 1
# For C_in=128, C_mid=256, k=1: 32768 vs 16512 — 50% reduction
```

**RepDWConv** uses three parallel branches during training:
- 3×3 depthwise conv (primary)  
- 1×1 depthwise conv (correction)
- Identity BN (residual)

All three fuse into a single 3×3 DWConv at inference time via `fuse_convs()`.

### Channel Scaling (Nano)

```
Channel counts at each stage output:
  Stem:   16  (stride 2)
  P2:     32  (stride 4) — high-res, crucial for small objects
  P3:     64  (stride 8)
  P4:    128  (stride 16)
  P5:    256  (stride 32) — low-res, semantic context
```

---

## 2. Neck: Adaptive BiFPN

### Weighted Feature Fusion

Standard feature pyramids concatenate or add features equally. BiFPN introduces **learnable per-channel weights** for each fusion:

```python
# Fast normalized fusion:
out = Conv( w1·P3 + w2·↑P4 ) / (|w1| + |w2| + ε)
```

where `w1, w2` are learnable scalars and `↑` denotes 2× nearest-neighbor upsampling.

### Architecture

```
                    P5_in (stride 32)
                      │
                      ▼ 1×1 projection
                    P5_proj
                      │
                      ▼
              ┌───────────────┐
              │               │
              ▼               │
          ┌────────┐          │
          │  P5_td │          │
          └───┬────┘          │
              │               │
         upsample▲2×          │
              │               │
        P4_in ┼───────────────┤
          │   │               │
          ▼   ▼               │
     ┌────────────┐           │
     │Weighted ×2 │           │
     │ P4 + ↑P5   │           │
     └──────┬─────┘           │
            │                 │
        Conv 3×3              │
            │                 │
       ┌────┴────┐            │
       │  P4_td  │            │
       └────┬────┘            │
            │                 │
       upsample▲2×            │
            │                 │
      P3_in ┼─────────────────┘
        │   │
        ▼   ▼
     ┌────────────┐
     │Weighted ×2 │
     │ P3 + ↑P4   │
     └──────┬─────┘
            │
        Conv 3×3
            │
       ┌────┴────┐
       │  P3_td  │
       └────┬────┘
            │   Top-down pass
            ▼   complete
         ┌──────┐
         │P3_out│ = P3_td
         └──┬───┘
            │
        downsample▼2×
            │
       P4_in┼──────────┬── P4_td
        │   │          │
        ▼   ▼          ▼
     ┌────────────┐
     │Weighted ×3 │
     │P4+P4_td+P3 │
     └──────┬─────┘
            │
        Conv 3×3
            │
       ┌────┴────┐
       │ P4_out  │
       └────┬────┘     Bottom-up pass
            │          complete
        downsample▼2×
            │
       P5_in┼──────────┬── P5_td
        │   │          │
        ▼   ▼          ▼
     ┌────────────┐
     │Weighted ×3 │
     │P5+P5_td+P4│
     └──────┬─────┘
            │
        Conv 3×3
            │
       ┌────┴────┐
       │ P5_out  │
       └─────────┘
```

**Key properties:**
- Only 3 weighted fusion weights per node (~3 scalar params) vs. full conv fusion
- Same structure for Nano and Small (channels differ)
- 1×1 projections unify backbone channels to BiFPN channel count

---

## 3. Head: LightEdge NMS-Free Decoupled Head

### Design Rationale

Standard YOLO heads produce dense predictions (e.g., 8400 per image at 640×640) that require NMS to filter duplicates. LightEdge uses a **one-to-one matching strategy**:

- Each ground-truth object is assigned to exactly **one** anchor during training
- During inference, top-k selection replaces NMS entirely
- Direct box regression (4 values: ltrb distances from anchor center) replaces DFL

### Decoupled Architecture

```
        Feature from BiFPN (C_bifpn, H, W)
                    │
          ┌─────────┴─────────┐
          │                   │
    ┌─────▼──────┐     ┌─────▼──────┐
    │  Box Branch │     │  Cls Branch│
    │             │     │            │
    │ Conv 3×3    │     │ Conv 3×3   │
    │ Conv 3×3    │     │ Conv 3×3   │
    │ Conv2d 1×1  │     │ Conv2d 1×1 │
    │ (→ 4 vals)  │     │ (→ nc vals)│
    └──────┬──────┘     └──────┬──────┘
           │                   │
           ▼                   ▼
     ltrb offsets        class logits
```

### Two-Branch Training (One2Many + One2One)

During training, two parallel heads exist:

| Head | Assignment | Top-K | Role |
|------|-----------|-------|------|
| **One2Many** | Task-aligned | 7 | Rich supervisory signal (standard YOLO) |
| **One2One** | Best-match | 1 | Clean NMS-free inference |

During inference, the One2Many head is removed (via `head.fuse()`) and only One2One is used.

### Anchor Generation

```python
# For each FPN level i with stride s_i and feature map (H_i, W_i):
#   cx = arange(W_i) + 0.5
#   cy = arange(H_i) + 0.5
#   anchors = stack(meshgrid(cx, cy))  # shape (H_i*W_i, 2)
#   strides = full((H_i*W_i, 1), s_i)
# Total anchors = sum(H_i * W_i) = 80*80 + 40*40 + 20*20 = 8400 @ 640px
```

### Box Decoding

```python
# Head predicts: [l, t, r, b] — distances from anchor center
# Decoded box:
#   x1 = cx - l     y1 = cy - t
#   x2 = cx + r     y2 = cy + b
# Apply stride scaling:
#   x1 *= stride    etc.
```

### Top-K Post-Processing (NMS replacement)

```python
# For each image:
#   1. Find top-300 scores across all classes
#   2. Gather corresponding boxes and class indices
#   Output: [x1, y1, x2, y2, score, class_id] × 300
```

---

## 4. Loss Functions

### Varifocal Loss (Classification)

```python
VFL(p, q) = Σ [ α · p^γ · (1 - q) · BCE(p, 0)   +   q · BCE(p, q) ]
                └─── negative weight ───┘   └─── positive weight ───┘
```

- **α = 0.75**, **γ = 2.0**
- For positive matches, target `q` = IoU between predicted and GT box (soft target)
- For negative matches, only BCE with focal-style down-weighting
- Naturally handles class imbalance better than BCE or Focal Loss

### GIoU Loss (Box Regression)

```
GIoU = IoU - (C - (A ∪ B)) / C    where C = convex hull
Loss = 1 - GIoU
```

- Only applied to matched positive anchors
- Provides gradient even when boxes don't overlap (unlike standard IoU)

### SmallObjectBoostLoss

```python
SmallObjectBoost(GIoU, gt_box) = GIoU · (1 + 2.0 · 1[sqrt(area) < 32])
```

- Doubles the GIoU loss weight for objects with sqrt(area) < 32 pixels
- Forces the model to allocate more capacity to small-object regression
- Critical for achieving competitive AP_S

### Combined Loss

```python
L = L_one2one + 0.5 · L_one2many
  where L_branch = VarifocalLoss + GIoULoss + SmallObjectBoost
```

The 0.5 weight on One2Many encourages the One2One head to be the primary prediction path.

---

## 5. Reparameterization

### RepDWConv Fusion

During training, `RepDWConv` maintains three parallel branches:
1. `Conv2d(c, c, 3, groups=c)` → `BN` (primary 3×3 DW)
2. `Conv2d(c, c, 1, groups=c)` → `BN` (1×1 correction)
3. `BN` (identity, only when stride=1)

At inference time, `fuse_convs()` collapses them into a single `Conv2d(c, c, 3, groups=c, bias=True)`:

```python
kernel = kernel_3x3 + pad(kernel_1x1) + identity_kernel
bias = bias_3x3 + bias_1x1 + bias_identity
```

This gives a **clean, branch-free compute graph** for inference — critical for TensorRT and CoreML.

### Conv-BN Fusion

Standard `fuse_conv_and_bn()` folds batch normalization into preceding `Conv2d` weights:

```
W_fused = W · γ / σ
b_fused = β - μ · γ / σ
```

All Conv layers with BN support `forward_fuse()` which skips BN.

---

## 6. Quantization Considerations

| Component | QAT-Friendly? | Notes |
|-----------|--------------|-------|
| GhostConv | ✅ | Only depthwise and 1×1 convs — well-behaved under INT8 |
| RepDWConv | ✅ | Single depthwise conv after fusion — excellent quantization |
| CoordAtt | ✅ | Sigmoid outputs in [0,1], pooling is integer-friendly |
| ECA | ✅ | Conv1d + sigmoid in [0,1], very low dynamic range |
| BiFPN | ⚠️ | Weighted fusion weights should be clipped to avoid overflow |
| Head | ✅ | No softmax in decode path (sigmoid for cls only) |
| SiLU | ⚠️ | Replace with ReLU or hard-Swish for INT8 deployment |

**Recommended QAT flow:**
1. Train FP32 model
2. Fuse RepDWConv branches
3. Replace SiLU with hard-Swish (ReLU6 variant)
4. Apply `torch.quantization` or `pytorch_quantization` for INT8 calibration
5. Validate accuracy drop (expected < 1% mAP)

---

## 7. Comparison with YOLO26n/s

| Aspect | YOLO26n | LightEdge-Nano | YOLO26s | LightEdge-Small |
|--------|---------|----------------|---------|-----------------|
| Params | 2.4M | 3.0M | 9.5M | 8.4M |
| FLOPs | 5.4B | 13.5B | 20.7B | 34.3B |
| NMS | Yes | No | Yes | No |
| DFL | Yes (reg_max=1) | No | Yes | No |
| Head Type | Coupled + one2one | Decoupled + one2one | Coupled + one2one | Decoupled + one2one |
| Backbone | C3k2 + SPPF | RepViTGhost + Attn | C3k2 + SPPF | RepViTGhost + Attn |
| Neck | FPN + PAN | Adaptive BiFPN | FPN + PAN | Adaptive BiFPN |
| Activations | SiLU | SiLU | SiLU | SiLU |
| Export | ONNX/TRT/CoreML | ONNX/TRT/CoreML/TFLite | ONNX/TRT/CoreML | ONNX/TRT/CoreML/TFLite |

**Trade-offs:**
- LightEdge has higher FLOPs due to attention modules and BiFPN, but removes NMS latency
- No DFL means simpler compute graph but potentially coarser box precision
- GhostConv + RepDWConv gives better parameter efficiency per FLOP
- CoordAtt + ECA adds accuracy without significant parameter overhead

---

## 8. Edge Deployment Guide

### Raspberry Pi 4/5

```bash
# Export to ONNX with FP32
python -c "
from ultralytics.nn.modules.lightedge import LightEdgeYOLO, export_onnx
model = LightEdgeYOLO('nano')
model.eval()
export_onnx(model, 'lightedge_nano.onnx', imgsz=640)
"

# Run with ONNX Runtime
python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('lightedge_nano.onnx')
input_name = sess.get_inputs()[0].name
output = sess.run(None, {input_name: np.random.randn(1, 3, 640, 640).astype(np.float32)})
print(output[0].shape)  # (1, 300, 6)
"
```

### Jetson Orin

```bash
# Export to TensorRT with FP16
python -c "
from ultralytics.nn.modules.lightedge import LightEdgeYOLO, export_tensorrt
model = LightEdgeYOLO('small')
model.eval()
export_tensorrt(model, 'lightedge_small.engine', imgsz=640, fp16=True)
"

# Run with TensorRT
trtexec --loadEngine=lightedge_small.engine --shapes=images:1x3x640x640
```

### Mobile (CoreML)

```bash
python -c "
from ultralytics.nn.modules.lightedge import LightEdgeYOLO, export_coreml
model = LightEdgeYOLO('nano')
model.eval()
export_coreml(model, 'lightedge_nano.mlpackage', imgsz=640)
"
```

---

## References

- [GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)
- [RepVGG: Making VGG-style ConvNets Great Again](https://arxiv.org/abs/2101.03697)
- [Coordinate Attention for Efficient Mobile Network Design](https://arxiv.org/abs/2103.02907)
- [ECA-Net: Efficient Channel Attention for Deep CNNs](https://arxiv.org/abs/1910.03151)
- [EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/abs/1911.09070)
- [VarifocalNet: An IoU-aware Dense Object Detector](https://arxiv.org/abs/2008.13367)
- [RepViT: Revisiting Mobile CNN From ViT Perspective](https://arxiv.org/abs/2307.09283)
