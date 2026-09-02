#!/usr/bin/env python3
"""
Grad-CAM visualization for LIGHTEDGEDET

Targets:
    - P3
    - P4
    - P5
    - P6
    - CrossScaleFusion
    - Any arbitrary layer

The script:
    1. Loads a LIGHTEDGEDET checkpoint
    2. Lists model layers if requested
    3. Registers forward/backward hooks
    4. Runs inference
    5. Selects a detection/class score
    6. Computes Grad-CAM
    7. Saves:
         - original image
         - raw heatmap
         - Grad-CAM overlay
         - side-by-side visualization

Usage examples:

    python gradcam_lightedgedet.py \
        --weights best.pt \
        --source image.jpg \
        --layer 150

    python gradcam_lightedgedet.py \
        --weights best.pt \
        --source image.jpg \
        --layer P3

    python gradcam_lightedgedet.py \
        --weights best.pt \
        --source image.jpg \
        --layer CrossScaleFusion

    python gradcam_lightedgedet.py \
        --weights best.pt \
        --source image.jpg \
        --list-layers
"""

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Grad-CAM for LIGHTEDGEDET"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to LIGHTEDGEDET checkpoint"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input image"
    )

    parser.add_argument(
        "--layer",
        type=str,
        default=None,
        help="Target layer index or layer name"
    )

    parser.add_argument(
        "--list-layers",
        action="store_true",
        help="Print all model layers and exit"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="cuda:0 / cpu"
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Target class ID. If omitted, highest scoring class is used."
    )

    parser.add_argument(
        "--detection-index",
        type=int,
        default=0,
        help="Detection location index used as target"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="gradcam_results",
        help="Output directory"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Heatmap overlay transparency"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Number of highest scoring predictions used for target"
    )

    return parser.parse_args()


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(weights, device):

    print("\n[INFO] Loading model...")

    device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )

    try:

        from ultralytics import YOLO

        yolo = YOLO(weights)

        model = yolo.model

        print("[INFO] Loaded using Ultralytics YOLO")

    except Exception as e:

        print("[WARNING] Ultralytics loading failed:")
        print(e)

        checkpoint = torch.load(
            weights,
            map_location=device,
            weights_only=False
        )

        if isinstance(checkpoint, dict):

            if "ema" in checkpoint:
                model = checkpoint["ema"]

            elif "model" in checkpoint:
                model = checkpoint["model"]

            else:
                raise RuntimeError(
                    "Cannot find model inside checkpoint."
                )

        else:
            model = checkpoint

    model = model.to(device)
    model.eval()

    return model, device


# ============================================================
# LAYER LIST
# ============================================================

def list_layers(model):

    print("\n")
    print("=" * 100)
    print("LIGHTEDGEDET MODEL LAYERS")
    print("=" * 100)

    for idx, (name, module) in enumerate(model.named_modules()):

        if len(list(module.children())) == 0:

            params = sum(
                p.numel()
                for p in module.parameters()
            )

            print(
                f"[{idx:4d}] "
                f"{name:60s} "
                f"{module.__class__.__name__:30s} "
                f"params={params:,}"
            )

    print("=" * 100)
    print()


# ============================================================
# FIND TARGET LAYER
# ============================================================

def find_layer(model, layer_string):

    modules = list(model.named_modules())

    # --------------------------------------------------------
    # Numeric index
    # --------------------------------------------------------

    try:

        index = int(layer_string)

        leaf_modules = []

        for name, module in modules:

            if len(list(module.children())) == 0:
                leaf_modules.append((name, module))

        if index < 0 or index >= len(leaf_modules):

            raise ValueError(
                f"Layer index {index} out of range. "
                f"Number of leaf layers = {len(leaf_modules)}"
            )

        name, module = leaf_modules[index]

        print(
            f"[INFO] Selected layer [{index}] "
            f"{name} -> {module.__class__.__name__}"
        )

        return module

    except ValueError:
        pass

    # --------------------------------------------------------
    # Exact name
    # --------------------------------------------------------

    for name, module in modules:

        if name == layer_string:

            print(
                f"[INFO] Selected layer "
                f"{name} -> {module.__class__.__name__}"
            )

            return module

    # --------------------------------------------------------
    # Partial name
    # --------------------------------------------------------

    matches = []

    query = layer_string.lower()

    for name, module in modules:

        text = (
            name + " " +
            module.__class__.__name__
        ).lower()

        if query in text:
            matches.append(
                (name, module)
            )

    if len(matches) == 1:

        name, module = matches[0]

        print(
            f"[INFO] Selected layer "
            f"{name} -> {module.__class__.__name__}"
        )

        return module

    if len(matches) > 1:

        print("\n[WARNING] Multiple matching layers:")

        for name, module in matches:

            print(
                f"  {name:60s} "
                f"{module.__class__.__name__}"
            )

        raise RuntimeError(
            "Please use an exact layer name."
        )

    raise RuntimeError(
        f"Cannot find target layer: {layer_string}"
    )


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.backward_hook
            )
        )

    # --------------------------------------------------------
    # Forward hook
    # --------------------------------------------------------

    def forward_hook(
        self,
        module,
        input,
        output
    ):

        if isinstance(output, (tuple, list)):

            output = output[0]

        self.activations = output

    # --------------------------------------------------------
    # Backward hook
    # --------------------------------------------------------

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        gradient = grad_output[0]

        if gradient is None:
            return

        self.gradients = gradient

    # --------------------------------------------------------
    # Generate CAM
    # --------------------------------------------------------

    def generate(self):

        if self.activations is None:

            raise RuntimeError(
                "No activation captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "No gradient captured."
            )

        activation = self.activations
        gradient = self.gradients

        # ----------------------------------------------------
        # Expected feature shape:
        #
        # [B, C, H, W]
        # ----------------------------------------------------

        if activation.ndim != 4:

            raise RuntimeError(
                f"Target layer output has shape "
                f"{activation.shape}. "
                f"Grad-CAM requires [B,C,H,W]."
            )

        # ----------------------------------------------------
        # Global average pooling of gradients
        # ----------------------------------------------------

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activation
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = cam[0, 0]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam -= cam.min()

        max_value = cam.max()

        if max_value > 0:

            cam /= max_value

        return cam.detach().cpu().numpy()

    # --------------------------------------------------------

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def letterbox(
    image,
    new_shape=640
):

    h, w = image.shape[:2]

    scale = min(
        new_shape / h,
        new_shape / w
    )

    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (new_shape, new_shape, 3),
        114,
        dtype=np.uint8
    )

    dw = (new_shape - new_w) // 2
    dh = (new_shape - new_h) // 2

    canvas[
        dh:dh + new_h,
        dw:dw + new_w
    ] = resized

    return canvas, scale, dw, dh


def preprocess(image, imgsz, device):

    image_lb, scale, dw, dh = letterbox(
        image,
        imgsz
    )

    rgb = cv2.cvtColor(
        image_lb,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    return tensor, image_lb, scale, dw, dh


# ============================================================
# OUTPUT EXTRACTION
# ============================================================

def flatten_outputs(output):

    tensors = []

    if torch.is_tensor(output):

        tensors.append(output)

    elif isinstance(output, (tuple, list)):

        for x in output:
            tensors.extend(
                flatten_outputs(x)
            )

    elif isinstance(output, dict):

        for x in output.values():
            tensors.extend(
                flatten_outputs(x)
            )

    return tensors


# ============================================================
# FIND DETECTION SCORE
# ============================================================

def get_detection_target(
    output,
    class_id=None,
    top_k=1
):

    tensors = flatten_outputs(output)

    if len(tensors) == 0:

        raise RuntimeError(
            "No tensor found in model output."
        )

    # --------------------------------------------------------
    # Find tensor with detection-like shape
    #
    # LIGHTEDGEDET:
    #
    # [B, 4+nc, 8500]
    #
    # --------------------------------------------------------

    candidate = None

    for tensor in tensors:

        if tensor.ndim == 3:

            # [B, C, N]
            if tensor.shape[1] >= 5:

                candidate = tensor
                break

            # [B, N, C]
            if tensor.shape[2] >= 5:

                candidate = tensor
                break

    if candidate is None:

        raise RuntimeError(
            "Could not identify detection output."
        )

    pred = candidate

    print(
        "[INFO] Detection output shape:",
        tuple(pred.shape)
    )

    # --------------------------------------------------------
    # Convert [B,C,N] -> [B,N,C]
    # --------------------------------------------------------

    if pred.shape[1] < pred.shape[2]:

        pred = pred.transpose(1, 2)

    # pred:
    #
    # [B, N, 4+nc]
    #

    if pred.shape[-1] <= 4:

        raise RuntimeError(
            "Detection tensor does not contain classes."
        )

    # --------------------------------------------------------
    # Classification scores
    #
    # reg_max=1
    #
    # output:
    # 4 + nc
    # --------------------------------------------------------

    cls_scores = pred[..., 4:]

    # --------------------------------------------------------
    # Specific class
    # --------------------------------------------------------

    if class_id is not None:

        if class_id >= cls_scores.shape[-1]:

            raise ValueError(
                f"class_id={class_id} but "
                f"number of classes="
                f"{cls_scores.shape[-1]}"
            )

        scores = cls_scores[
            0,
            :,
            class_id
        ]

        values, indices = torch.topk(
            scores,
            k=min(
                top_k,
                scores.shape[0]
            )
        )

        target = values.sum()

        print(
            f"[INFO] Target class: {class_id}"
        )

    # --------------------------------------------------------
    # Automatically choose strongest class
    # --------------------------------------------------------

    else:

        max_scores, class_indices = (
            cls_scores[0].max(dim=-1)
        )

        values, indices = torch.topk(
            max_scores,
            k=min(
                top_k,
                max_scores.shape[0]
            )
        )

        target = values.sum()

        selected_classes = (
            class_indices[indices]
            .detach()
            .cpu()
            .tolist()
        )

        print(
            "[INFO] Selected classes:",
            selected_classes
        )

    print(
        "[INFO] Target score:",
        float(target.detach().cpu())
    )

    return target


# ============================================================
# CREATE HEATMAP
# ============================================================

def create_heatmap(
    cam,
    original_shape
):

    h, w = original_shape[:2]

    cam = cv2.resize(
        cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    cam_uint8 = np.uint8(
        255 * cam
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    return heatmap, cam


# ============================================================
# OVERLAY
# ============================================================

def overlay_heatmap(
    image,
    heatmap,
    alpha=0.45
):

    overlay = cv2.addWeighted(
        image,
        1.0 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


# ============================================================
# SAVE SIDE-BY-SIDE
# ============================================================

def save_comparison(
    original,
    heatmap,
    overlay,
    path
):

    h, w = original.shape[:2]

    heatmap = cv2.resize(
        heatmap,
        (w, h)
    )

    overlay = cv2.resize(
        overlay,
        (w, h)
    )

    comparison = np.concatenate(
        [
            original,
            heatmap,
            overlay
        ],
        axis=1
    )

    cv2.imwrite(
        path,
        comparison
    )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    os.makedirs(
        args.output,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model, device = load_model(
        args.weights,
        args.device
    )

    # --------------------------------------------------------
    # List layers
    # --------------------------------------------------------

    if args.list_layers:

        list_layers(model)

        return

    if args.layer is None:

        print(
            "\nERROR: --layer is required."
        )

        print(
            "\nFirst run:"
        )

        print(
            "python gradcam_lightedgedet.py "
            "--weights best.pt "
            "--source image.jpg "
            "--list-layers"
        )

        return

    # --------------------------------------------------------
    # Find target layer
    # --------------------------------------------------------

    target_layer = find_layer(
        model,
        args.layer
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    gradcam = GradCAM(
        model,
        target_layer
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        args.source
    )

    if image is None:

        raise FileNotFoundError(
            f"Cannot read image: {args.source}"
        )

    original = image.copy()

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor, _, _, _, _ = preprocess(
        image,
        args.imgsz,
        device
    )

    # --------------------------------------------------------
    # Enable gradients
    # --------------------------------------------------------

    input_tensor.requires_grad_(True)

    model.zero_grad(
        set_to_none=True
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    print("\n[INFO] Forward pass...")

    output = model(
        input_tensor
    )

    # --------------------------------------------------------
    # Detection target
    # --------------------------------------------------------

    target = get_detection_target(
        output,
        class_id=args.class_id,
        top_k=args.top_k
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    print(
        "[INFO] Backpropagating..."
    )

    target.backward(
        retain_graph=False
    )

    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    cam = gradcam.generate()

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap, cam_resized = create_heatmap(
        cam,
        original.shape
    )

    overlay = overlay_heatmap(
        original,
        heatmap,
        args.alpha
    )

    # --------------------------------------------------------
    # Output names
    # --------------------------------------------------------

    layer_name = args.layer.replace(
        "/",
        "_"
    ).replace(
        ".",
        "_"
    )

    base = Path(
        args.source
    ).stem

    prefix = os.path.join(
        args.output,
        f"{base}_{layer_name}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    cv2.imwrite(
        prefix + "_original.jpg",
        original
    )

    cv2.imwrite(
        prefix + "_heatmap.jpg",
        heatmap
    )

    cv2.imwrite(
        prefix + "_gradcam.jpg",
        overlay
    )

    save_comparison(
        original,
        heatmap,
        overlay,
        prefix + "_comparison.jpg"
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    gradcam.remove_hooks()

    print("\n" + "=" * 70)
    print("Grad-CAM completed")
    print("=" * 70)

    print(
        "Original :",
        prefix + "_original.jpg"
    )

    print(
        "Heatmap  :",
        prefix + "_heatmap.jpg"
    )

    print(
        "Grad-CAM :",
        prefix + "_gradcam.jpg"
    )

    print(
        "Compare  :",
        prefix + "_comparison.jpg"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()