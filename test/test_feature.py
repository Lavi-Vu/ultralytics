import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import math

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolo11n.pt"
IMAGE_PATH = "/home/lavi/Pictures/Screenshots/Screenshot from 2026-04-06 15-59-05.png"

OUTPUT_DIR = "feature_outputs"
GRID_PATH = os.path.join(OUTPUT_DIR, "paper_style_heatmap.png")
LAYER_DIR = os.path.join(OUTPUT_DIR, "layers")
FEATURE_DIR = os.path.join(OUTPUT_DIR, "feature_maps")

IMG_SIZE = 640
DISPLAY_SIZE = 256   # 🔥 smaller map → bigger visualization
COLS = 6

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LAYER_DIR, exist_ok=True)
os.makedirs(FEATURE_DIR, exist_ok=True)

# =========================
# LOAD MODEL
# =========================
model = YOLO(MODEL_PATH)
model.model.eval()

# =========================
# LOAD IMAGE
# =========================
img = cv2.imread(IMAGE_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

img_tensor = torch.from_numpy(img).float() / 255.0
img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

# =========================
# HOOK ALL LAYERS
# =========================
features = []

def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        features.append(output.detach().cpu())

hooks = []
for layer in model.model.model:
    hooks.append(layer.register_forward_hook(hook_fn))

# =========================
# FORWARD PASS
# =========================
with torch.no_grad():
    _ = model.model(img_tensor)

for h in hooks:
    h.remove()

print(f"Captured {len(features)} layers")

# =========================
# SAVE FEATURE MAP CHANNELS (FLAT)
# =========================
MAX_CHANNELS = 8  # avoid explosion

for layer_idx, feat in enumerate(features):
    feat = feat[0]  # [C,H,W]
    C = feat.shape[0]

    num_show = min(C, MAX_CHANNELS)

    for ch in range(num_show):
        fmap = feat[ch].numpy()

        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
        fmap = (fmap * 255).astype(np.uint8)

        fmap = cv2.resize(fmap, (DISPLAY_SIZE, DISPLAY_SIZE))

        save_path = os.path.join(FEATURE_DIR, f"layer{layer_idx:03d}_ch{ch:02d}.png")
        cv2.imwrite(save_path, fmap)

print(f"✅ Feature maps saved to: {FEATURE_DIR}")

# =========================
# PROCESS + SAVE EACH LAYER (HEATMAP OVERLAY)
# =========================
overlay_maps = []

for i, feat in enumerate(features):
    feat = feat[0]

    # 🔥 use max activation (sharper)
    fmap = torch.max(feat, dim=0)[0].numpy()

    fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-6)
    fmap = cv2.resize(fmap, (DISPLAY_SIZE, DISPLAY_SIZE)) 

    # ---- warm heatmap (like your image) ----
    heatmap = (fmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_INFERNO)

    # resize original to match
    img_small = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE))

    # overlay
    overlay = cv2.addWeighted(img_small, 0.6, heatmap, 0.4, 0)

    overlay_maps.append(overlay)

    # save each layer
    save_path = os.path.join(LAYER_DIR, f"layer_{i:03d}.png")
    cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

print(f"✅ Saved individual layers to: {LAYER_DIR}")

# =========================
# CREATE GRID (PAPER STYLE)
# =========================
total = len(overlay_maps) + 1
rows = math.ceil(total / COLS)

plt.figure(figsize=(COLS * 4, rows * 4))  # 🔥 bigger tiles

# Input image
plt.subplot(rows, COLS, 1)
plt.imshow(cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE)))
plt.title("Input")
plt.axis('off')

# Feature overlays
for i, overlay in enumerate(overlay_maps):
    plt.subplot(rows, COLS, i + 2)
    plt.imshow(overlay)
    plt.title(f"L{i}")
    plt.axis('off')

plt.tight_layout()
plt.savefig(GRID_PATH, dpi=300)
plt.show()

print(f"✅ Grid saved to: {GRID_PATH}")