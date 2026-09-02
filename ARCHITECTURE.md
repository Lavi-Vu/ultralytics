# LightEdgeDet — Architecture

LightEdgeDet is a custom, resource-efficient object detector built on the
Ultralytics fork. It combines a **CNN + split-route-attention hybrid backbone**
with a **lightweight reparameterized FPN neck** and a **decoupled detection head**,
targeting edge/consumer devices (cameras, robots, on-device assistants) where
accuracy and FLOP/parameter budget must both be respected.

This document describes the architecture at a level that lets a reader
reconstruct the model from the configs and understand the design decisions
behind each component. All referenced source files live under
`ultralytics/`.

---

## 1. Top-level dataflow

```
 input RGB (e.g. 640×640)
      │
      ▼
┌─────────────┐
│  Backbone   │   HybridBackbone — 5 stages, stem + SPPF
│             │   outputs [P3, P4, P5, P6]  (strides 8/16/32/64)
└─────────────┘
      │ multi-scale features
      ▼
┌─────────────┐
│    Neck     │   LitePAFPN — lateral + FPN(top-down) + PAN(bottom-up)
│             │   + CrossScaleFusion global gate
└─────────────┘
      │ 4 levels @ neck_out_channels
      ▼
┌─────────────┐
│    Head     │   Detect (standard) or LightDetectHead (shared-cls)
│             │   → [B, 4+nc, 8500] raw tensor at 640 px
└─────────────┘
```

The `LightEdgeDetModel` in `ultralytics/nn/tasks.py` builds the three parts
directly from flat YAML keys (it does **not** go through `parse_model`).
`predict` is simply `backbone → neck → detect`.

---

## 2. Backbone — `HybridBackbone`

Source: `ultralytics/nn/modules/lightedgedet.py` (`class HybridBackbone`, line 230)

The backbone extracts a 4-level feature pyramid over 5 stages. It combines
depthwise-separable CNN blocks (`LiteBlock`) with optional lightweight
split-route attention (`LitePSA`) placed in selected deep stages, plus a cheap
large-kernel-depthwise recipe at low resolution.

### 2.1 Stem

A single `3×3` stride-2 convolution (bias-free) + BatchNorm + SiLU, halving the
input resolution (640 → 320 feature plane):

```python
self.stem = nn.Sequential(
    nn.Conv2d(c1, stem_channels, 3, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(stem_channels),
    nn.SiLU(inplace=True),
)
```

### 2.2 Stage structure

There are 5 stages (`backbone_depths`), each a `nn.Sequential` of `LiteBlock`s
(optionally swapped for `LitePSA` at the middle position):

- The **first block of each stage downsamples** by stride 2 (no residual).
- Later blocks are stride 1 with a residual connection.
- When `attn_stages[i] == 1`, the **middle block** (`bi == depth // 2`) of
  stage `i` is replaced by a `LitePSA`.
- Drop-Path rates are linearly interpolated across the whole model via
  `torch.linspace(0, drop_path_rate, sum(depths))`.
- Depthwise kernels come from `backbone_kernel_sizes` — deep stages use
  **7×7 large-kernel depthwise** (HGNetv2/PResNet recipe), shallow stages 3×3.

### 2.3 Output pyramid + SPPF

```python
self.out_indices = [1, 2, 3, 4]          # collect stage outputs
self.sppf = SPPF(channels_list[-1], channels_list[-1], k=5) if use_sppf else nn.Identity()

def forward(self, x):
    features = []
    x = self.stem(x)
    for idx, stage in enumerate(self.stages):
        x = stage(x)
        if idx in self.out_indices:
            features.append(x)
    if isinstance(self.sppf, SPPF):
        features[-1] = self.sppf(features[-1])
    return features
```

The four pyramid levels P3–P6 map to stages 1–4 with strides 8/16/32/64. The
deepest level (P6) is passed through an SPPF module (max-pooling receptive
fields 5×5 → 9×9 → 13×13) before the neck.

### 2.4 `LiteBlock` — core CNN block

Source: `lightedgedet.py` (`class LiteBlock`, line 128)

An MBConv-style depthwise-separable residual block:

```
[1×1 expand -> c1→mid_c] → [DW k×k] → SiLU → [SE] → [1×1 project] → (+ residual)
```

where `mid_c = int(c1 * expand_ratio)`. Details:

- **Expand** (`Conv(c1, mid_c, 1)`): 1×1 + BN + SiLU; identity when
  `expand_ratio == 1.0`.
- **Depthwise**: `RepDWConv(mid_c, k, 1)` when `reparam and stride == 1`,
  otherwise a plain grouped `Conv(mid_c, mid_c, k, stride, g=mid_c)`.
- **SE**: `SqueezeExcitation(mid_c, reduction=4)` when `se_ratio > 0`.
- **Project** (`Conv(mid_c, c2, 1)`): 1×1, no activation.
- **Residual + DropPath** applied when `stride == 1 and c1 == c2`.

```python
def forward(self, x):
    h = self.dw(self.expand(x))
    h = F.silu(h, inplace=True)
    h = self.proj(self.se(h))
    return self.drop_path(h) + x if self.use_residual else h
```

### 2.5 `RepDWConv` — reparameterized depthwise conv

Source: `lightedgedet.py` (`class RepDWConv`, line 62)

During **training** a stride-1 `RepDWConv` is the sum of up to three branches:

```python
y = self.conv3(x)          # k×k depthwise + BN
y = y + self.conv1(x)      # 1×1 depthwise + BN   (only if stride == 1)
y = y + x                  # identity              (only if stride == 1)
```

All three branches operate at the same channel width and resolution, so they
fold **exactly** at inference via `fuse()`: the 1×1 kernel is written into the
center tap of the k×k weight, the identity is added as a +1.0 center value, and
both BNs are absorbed. The result is a single plain depthwise convolution
(`forward_fuse`). This yields **dense-like training capacity at plain-depthwise
inference cost** (RepVGG-style).

### 2.6 `LitePSA` — split-route attention

Source: `lightedgedet.py` (`class LitePSA`, line 190)

A C2PSA-style split-route block. The hidden width is halved into two routes:

```python
c_ = max(int(c * e), 8)                    # e = psa_ratio = 0.5
self.cv1  = Conv(c, 2 * c_, 1)
self.local = nn.Sequential(Conv(c_, c_, 3, 1, 1, g=c_), Conv(c_, c_, 1, act=False))
self.attn  = nn.Sequential(*(PSABlock(c_) for _ in range(n)))   # context route
self.cv2   = Conv(2 * c_, c, 1, act=False)

def forward(self, x):
    a, b = self.cv1(x).split(...)
    a = self.local(a)      # local spatial detail
    b = self.attn(b)       # global context (PSA stack)
    return self.cv2(torch.cat((a, b), 1))
```

- **Local route**: 3×3 depthwise + 1×1 pointwise (preserves fine detail).
- **Context route**: a stack of `PSABlock`s (conv-based multi-head
  self-attention + feed-forward network, from `ultralytics/nn/modules/block.py`).

Because `LitePSA` sits at the middle, same-resolution position of a deep
low-resolution stage (P5/P6) and runs on a **halved** hidden width, the
quadratic attention term stays cheap. **Note:** unlike the older
`docs/ARCHITECTURE.md`, the attention `qkv` here uses plain `Conv` (BN-based);
the only `LayerNorm` in the whole network is inside `CrossScaleFusion`.

---

## 3. Neck — `LitePAFPN`

Source: `ultralytics/nn/modules/lightedgedet.py` (`class LitePAFPN`, line 398)

A lightweight FPN with top-down and bottom-up paths built entirely from
depthwise-separable blocks, followed by a global **CrossScaleFusion** gate.
The submodules are built lazily on the first forward (`_lazy_build`) once the
real input channels from the backbone are known.

```python
self.lateral_convs = ModuleList(...)   # 1×1 + BN + SiLU per level (Identity if c==c2)
self.fpn_convs     = ModuleList([_blk × num_blocks per level])   # top-down
self.pan_convs     = ModuleList([_blk × num_blocks per level])   # bottom-up
self.down_convs    = ModuleList([_LiteDownBlock × (num_levels-1)])  # stride-2
self.fusion        = CrossScaleFusion(c2) if use_cross_fusion else Identity
```

Forward:

```python
reduced = [lat(f) for lat, f in zip(self.lateral_convs, inputs)]

# top-down FPN
laterals = [self.fpn_convs[-1](reduced[-1])]
for i in range(num_levels-2, -1, -1):
    up = F.interpolate(laterals[-1], size=reduced[i].shape[2:], mode="nearest")
    laterals.append(self.fpn_convs[i](reduced[i] + up))
laterals = laterals[::-1]

# bottom-up PAN
outputs = [self.pan_convs[0](laterals[0])]
for i in range(1, num_levels):
    down = self.down_convs[i-1](outputs[-1])
    outputs.append(self.pan_convs[i](laterals[i] + down))

outputs = self.fusion(outputs)
```

- **`_LiteConvBlock`**: `RepDWConv(3×3) → SiLU → 1×1 → BN → SiLU` with residual
  (`conv(x) + x`), the depthwise analogue of RepNCSPELAN4's VGGBlock.
- **`_LiteDownBlock`**: SCDown-style learnable stride-2 downsampling —
  `5×5 depthwise s2 → BN → SiLU → 1×1 → BN → SiLU`.
- **Lateral** projections reduce each level to `neck_out_channels`.

### 3.1 `CrossScaleFusion` — global contextual gate

Source: `lightedgedet.py` (`class CrossScaleFusion`, line 360)

A compact distillation of RT-DETRv4-style cross-scale attention into a global
channel gate:

```python
self.cv   = nn.Conv2d(num_levels * c, c, 1, bias=False)   # fuse stacked scales
self.norm = nn.LayerNorm(c)
self.attn = nn.MultiheadAttention(c, num_heads, batch_first=True, bias=False)
self.ffn  = nn.Sequential(Conv2d(c, 2c, 1), SiLU, Conv2d(2c, c, 1))
self.gate = nn.Sequential(Conv2d(c, c, 1), SiLU, Conv2d(c, c, 1), Sigmoid)
```

Mechanics (see `forward`, line 383):

1. All four pyramid levels are bilinearly upsampled to the **lowest-resolution
   P6 size** and channel-concatenated; a 1×1 conv fuses them into one map `c`.
2. The fused map becomes a token sequence `(B, H·W, c)`; **LayerNorm → multi-head
   self-attention** → reshape back. A pre-norm residual then a FFN
   (`1×1 → c→2c → SiLU → 1×1`). The whole attention/FFN block lives at the
   cheapest resolution (`10×10` at 640 px).
3. A **global channel gate** is extracted via GAP → `1×1 → SiLU → 1×1 → Sigmoid`,
   and this single `(B, c, 1, 1)` vector is **multiplied into every level's
   feature map**:

```python
return [f * gate for f in feats]
```

So every scale (including small-object P3 features) receives an image-specific,
multi-scale-informed channel recalibration — at negligible additional FLOPs.

---

## 4. Detection head

Source: `ultralytics/nn/modules/head.py` (`class Detect`) and
`ultralytics/nn/modules/lightedgedet.py` (`class LightDetectHead`, line 485).

Two head options are selected via `head_type` in the YAML:

| `head_type` | Class | Description |
|---|---|---|
| `standard` | `Detect` | Standard per-scale head (YOLO26 end-to-end recipe) |
| `shared` | `LightDetectHead` | Decoupled, **shared classification** + per-level regression |

### 4.1 Standard `Detect`

Per-level regression (`cv2`) and classification (`cv3`) branches, decoded to an
anchor-free center + ltrb representation:

```python
# head.py forward → _inference
dbox = self._get_decode_boxes(x)          # center + ltrb distances × stride
y = torch.cat((dbox, x["scores"].sigmoid()), 1)
```

With `reg_max == 1` the DFL module is `nn.Identity()`, so the 4 box channels are
used directly (no distribution decoding); regression is supervised with CIoU +
size-normalized L1. Anchors are generated per level as cell-center points; the
total at 640 px is 80² + 40² + 20² + 10² = **8500**.

### 4.2 `LightDetectHead` (shared-cls / decoupled)

A slimmer head that shares classification across all strides while keeping
regression **per-level**:

```python
self.reduce = ModuleList(Conv2d(ci -> head_channels) or Identity per level)
def _branch(out):       # slim branch
    return nn.Sequential(
        nn.Sequential(DWConv(c, c, 3), Conv(c, c, 1)),
        nn.Sequential(DWConv(c, c, 3), Conv(c, c, 1)),
        nn.Conv2d(c, out, 1),
    )
self.cv2 = ModuleList([_branch(4 * reg_max) for _ in range(num_levels)])  # PER-LEVEL reg
self.cv3 = ModuleList([_branch(nc)])                                      # ONE shared cls
```

`forward_head` exploits the length-1 cls list:

```python
scores = torch.cat([cls_head[0](reduce[i](x[i])) for i in range(num_levels)], dim=-1)
boxes  = torch.cat([box_head[i](reduce[i](x[i]))   for i in range(num_levels)], dim=-1)
```

- **Regression is per-level** because the task-aligned assigner scores anchors
  by `alignment = cls^α · IoU^β`; a box branch shared across all strides cannot
  regress reliably, keeping IoU low and starving the classifier.
- **Classification is one shared branch** applied to every level, so it is
  computed once rather than once-per-level → head FLOPs drop.

### 4.3 Bias initialization

Focal-loss prior: box bias = 2.0; cls bias per level =
`log(5 / nc / (640 / stride_i)²)` targeting ~0.01 object prior per scale.

---

## 5. Model construction — `LightEdgeDetModel`

Source: `ultralytics/nn/tasks.py` (`class LightEdgeDetModel`, line 1037),
`ultralytics/models/yolo/model.py` (task registration).

- `guess_model_task` special-cases LightEdgeDet when `backbone_type` or
  `neck_out_channels` is present in a YAML → task `"lightedgedet"`.
- `LightEdgeDetModel` builds the three parts **directly from flat YAML keys**
  (no `parse_model` layer list):

```python
# backbone
self.backbone = HybridBackbone(c1=ch, ..., channels_list=..., depths=...,
                                expand_ratios=..., se_ratios=..., attn_stages=...,
                                kernel_sizes=..., stem_channels=...)
# neck
self.neck = LitePAFPN(c1=1, c2=neck_out, num_blocks=..., use_cross_fusion=..., reparam=...)
# head
head = Detect(...)        if head_type == "standard"
head = LightDetectHead(...) otherwise
self.model = nn.Sequential(); self.model.add_module("detect", head)
```

- **Compound scaling**: supported via `scales[name] = [depth, width, max_ch]`.
- **Stride discovery**: a dummy forward recovers feature-map sizes and sets
  `self.detect.stride`.
- **Loss**: `E2ELoss` if `end2end`, else `v8DetectionLoss`.
- **`fuse()`** is overridden to walk **every** module (the inherited version only
  walks `self.model`, which holds just the head): it folds Conv/DWConv+BN, calls
  `RepDWConv.fuse()` / `LiteBlock.fuse()`, and **skips `Detect.fuse()` when not
  E2E** so the one2many branch survives for NMS inference.

---

## 6. Config reference

Configs live in `ultralytics/cfg/models/lightedgedet_*.yaml`. The table below
summarizes the family. All variants share depths `[2, 3, 4, 3, 2]`, expansion
`[2.5, 2.5, 3.0, 3.5, 4.0]`, SE `0.25`, drop-path `0.1`, SPPF on,
`psa_ratio=0.5`, `psa_blocks=1`, reparam on, `reg_max=1`, `strides
[8,16,32,64]`.

### 6.1 Backbone channel / attention / kernel profile

| Variant | Stem | Stage0(unused) | **P3** | **P4** | **P5** | **P6** | attn_stages | kernels |
|---|---|---|---|---|---|---|---|---|
| `nano` | 20 | 37 | 74 | 140 | 196 | 244 | `[0,0,0,1,1]` | `[3,3,3,7,7]` |
| `small` | 36 | 56 | 142 | 216 | 368 | 432 | `[0,0,0,0,1]` | `[3,3,3,7,7]` |
| `small_v2` | 36 | 56 | 142 | 216 | 368 | 432 | `[0,1,1,1,1]` | `[3,3,3,7,7]` |
| `medium` | 28 | 56 | 112 | 208 | 288 | 352 | `[0,0,0,1,1]` | `[3,3,3,7,7]` |
| `large` | 32 | 72 | 144 | 272 | 368 | 432 | `[0,0,0,1,1]` | `[3,3,3,7,7]` |

Only stages 1–4 (P3–P6) are collected; stage 0 width is internal only. When
`attn_stages[i] == 1`, the middle LiteBlock of stage `i` is replaced by LitePSA.

### 6.2 Neck / head / budget (fused inference, 640×640 unless noted)

| Variant | Neck (ch × blocks) | Head | Params | GFLOPs |
|---|---|---|---|---|
| `nano` | 104 × 3 | standard | 3.73M | 7.02 |
| `small` | 196 × 4 | standard | 13.20M | 22.31 |
| `small_v2` | 196 × 4 | standard | 11.51M | 21.34 |
| `medium` | 160 × 4 | standard | 8.30M | 15.90 |
| `large` | 192 × 4 | standard | 13.09M | 24.50 |
| `nano_slim` | 80 × 2 | shared (56) | 3.27M | 5.36 |
| `nano_slim_trim` | 80 × 2 | shared (56) | 2.58M | 4.54 |
| `nano_slim_agg` | 64 × 2 | shared (48) | 2.47M | 4.18 |
| `nano_e2e` | 104 × 3 | standard, end2end | 3.73M | 6.82 |

> **Note:** verify per-variant header comments — some (e.g. `small`, `small_v2`)
> carry stale parameter/FLOP figures that no longer match the measured model.

### 6.3 Component parameter split (measured, Nano/nano_slim)

| Variant | Backbone | Neck | Head |
|---|---|---|---|
| `nano` | 2.946M (78%) | 0.562M (15%) | 0.253M (7%) |
| `nano_slim` | 2.946M (89%) | 0.292M (9%) | 0.062M (2%) |

---

## 7. Key design ideas (summary)

1. **Hybrid efficiency** — depthwise-separable CNN blocks for the bulk, targeted
   attention only where it is cheap (deep, low-resolution stages, halved width).
2. **Reparameterization** — multi-branch training capacity folds exactly to a
   single depthwise conv at inference (RepDWConv, used in both backbone and
   neck blocks).
3. **Large-kernel depthwise** on deep stages (7×7) for a larger receptive field
   with no dense-FLOP blow-up (HGNetv2/PResNet recipe).
4. **Global context for all scales** — CrossScaleFusion broadcasts a global,
   image-specific channel gate to every pyramid level, including P3 small-object
   features, at negligible cost (attention over only 10×10 tokens).
5. **Decoupled, frugal head** — shared classification across strides, per-level
   regression to keep the task-aligned assigner healthy; `reg_max=1` avoids DFL.
6. **NMS-free option** — an E2E variant removes NMS, enabling lower latency
   deployments.

---

## 8. Source file map

| Component | Location |
|---|---|
| `HybridBackbone` (stem, stages, SPPF) | `ultralytics/nn/modules/lightedgedet.py:230` |
| `LiteBlock` | `lightedgedet.py:128` |
| `RepDWConv` | `lightedgedet.py:62` |
| `LitePSA` | `lightedgedet.py:190` |
| `CrossScaleFusion` | `lightedgedet.py:360` |
| `LitePAFPN` (+ `_LiteConvBlock`, `_LiteDownBlock`) | `lightedgedet.py:398`, `318`, `346` |
| `LightDetectHead` | `lightedgedet.py:485` |
| `LightEdgeDetModel` (construction, fuse, predict) | `ultralytics/nn/tasks.py:1037` |
| `Detect` forward / decode | `ultralytics/nn/modules/head.py:158` |
| `DFL`, `PSABlock`, `SPPF` | `ultralytics/nn/modules/block.py` |
| `make_anchors`, `dist2bbox` | `ultralytics/utils/tal.py` |
| Config family | `ultralytics/cfg/models/lightedgedet_*.yaml` |
