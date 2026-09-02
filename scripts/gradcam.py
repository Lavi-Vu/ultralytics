#!/usr/bin/env python3

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from pathlib import Path
from ultralytics import YOLO


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
        required=True
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True
    )

    parser.add_argument(
        "--target",
        type=str,
        default="p3",
        choices=[
            "p3",
            "p5",
            "op3",
            "op5"
        ]
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="gradcam",
        choices=[
            "gradcam",
            "features"
        ],
        help="gradcam = gradient-weighted heatmap; "
             "features = visualize dense feature maps"
    )

    parser.add_argument(
        "--top-channels",
        type=int,
        default=16,
        help="[features mode] number of top channels "
             "to visualize per layer (by L2 magnitude)"
    )

    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="[features mode] comma-separated layer "
             "names, e.g. backbone.stages.1,neck.fusion.cv"
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=640
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0"
    )

    parser.add_argument(
        "--class-id",
        type=int,
        default=None
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45
    )

    parser.add_argument(
        "--output",
        type=str,
        default="gradcam_results"
    )

    return parser.parse_args()


# ============================================================
# GLOBAL STORAGE
# ============================================================

class FeatureHook:

    def __init__(self, module, target_name):

        self.module = module
        self.target_name = target_name

        self.activation = None
        self.gradient = None

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # We only use a forward hook here.
        #
        # CrossScaleFusion returns a LIST:
        #
        # [
        #     O-P3,
        #     O-P4,
        #     O-P5,
        #     O-P6
        # ]
        #
        # Therefore register_full_backward_hook() on the
        # whole module cannot reliably capture the gradient.
        # ----------------------------------------------------

        self.forward_handle = module.register_forward_hook(
            self.forward_hook
        )

        self.tensor_hook_handle = None

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def forward_hook(
        self,
        module,
        inputs,
        output
    ):

        # ----------------------------------------------------
        # CrossScaleFusion
        # ----------------------------------------------------

        if self.target_name == "op3":

            # neck.fusion.cv returns a single tensor
            if torch.is_tensor(output):
                self.activation = output
            elif isinstance(output, (list, tuple)):
                self.activation = output[0]
            else:
                raise RuntimeError(
                    "Unexpected output type "
                    f"{type(output)}"
                )

        elif self.target_name == "op5":

            # neck.fusion.cv returns a single tensor
            if torch.is_tensor(output):
                self.activation = output
            elif isinstance(output, (list, tuple)):
                self.activation = output[2]
            else:
                raise RuntimeError(
                    "Unexpected output type "
                    f"{type(output)}"
                )

        # ----------------------------------------------------
        # Normal Tensor output
        # ----------------------------------------------------

        else:

            if isinstance(output, (list, tuple)):

                self.activation = output[0]

            else:

                self.activation = output

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Register gradient hook directly on the selected
        # Tensor.
        #
        # This works even though CrossScaleFusion itself
        # returns a list.
        # ----------------------------------------------------

        if not torch.is_tensor(self.activation):

            raise RuntimeError(
                f"Selected activation is not a Tensor: "
                f"{type(self.activation)}"
            )

        if not self.activation.requires_grad:

            raise RuntimeError(
                "Selected activation does not require "
                "gradient."
            )

        # retain_grad() so .pop() is available after
        # backward on non-leaf tensors.  register_hook
        # alone is unreliable when the tensor is consumed
        # by multiple downstream ops (PAN, Fusion, etc.).
        self.activation.retain_grad()

        self.tensor_hook_handle = None

    # ========================================================
    # SAVE GRADIENT
    # ========================================================

    def save_gradient(self, gradient):

        self.gradient = gradient

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self):

        self.forward_handle.remove()

        if self.tensor_hook_handle is not None:

            self.tensor_hook_handle.remove()

# ============================================================
# GET TARGET MODULE
# ============================================================

def get_target_module(model, target):

    # --------------------------------------------------------
    # Ultralytics model
    # --------------------------------------------------------

    root = model.model

    modules = dict(root.named_modules())

    if target == "p3":

        name = "backbone.stages.1"

    elif target == "p5":

        name = "backbone.stages.3"

    elif target == "op3":

        name = "neck.fpn_convs.0"

    elif target == "op5":

        name = "neck.pan_convs.2"

    else:

        raise ValueError(
            f"Unknown target: {target}"
        )

    if name not in modules:

        print("\nAvailable modules containing keyword:")

        keyword = (
            "backbone.stages"
            if target in ["p3", "p5"]
            else "neck.fusion"
        )

        for n in modules:

            if keyword in n:

                print(n)

        raise RuntimeError(
            f"Cannot find module: {name}"
        )

    print(
        f"[INFO] Target = {target}"
    )

    print(
        f"[INFO] Module = {name}"
    )

    print(
        f"[INFO] Type = "
        f"{modules[name].__class__.__name__}"
    )

    return modules[name]


# ============================================================
# PREPROCESS
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

    nw = int(round(w * scale))
    nh = int(round(h * scale))

    resized = cv2.resize(
        image,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (new_shape, new_shape, 3),
        114,
        dtype=np.uint8
    )

    dw = (new_shape - nw) // 2
    dh = (new_shape - nh) // 2

    canvas[
        dh:dh + nh,
        dw:dw + nw
    ] = resized

    return canvas


def preprocess(
    image,
    imgsz,
    device
):

    image = letterbox(
        image,
        imgsz
    )

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    tensor = torch.from_numpy(
        rgb
    ).float()

    tensor = tensor.permute(
        2, 0, 1
    )

    tensor = tensor / 255.0

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    return tensor


# ============================================================
# FIND DETECTION SCORE
# ============================================================

def extract_detection_tensor(output):

    """
    LIGHTEDGEDET output before post-processing:

        [B, 84, 8500]

    where:

        4 = bbox
        nc = 80
    """

    if torch.is_tensor(output):

        if output.ndim == 3:

            return output

    if isinstance(output, (tuple, list)):

        for x in output:

            result = extract_detection_tensor(x)

            if result is not None:
                return result

    if isinstance(output, dict):

        for x in output.values():

            result = extract_detection_tensor(x)

            if result is not None:
                return result

    return None


# ============================================================
# DENSE FEATURE VISUALIZATION
# ============================================================

def visualize_features(
    model,
    inp,
    args
):
    """Visualize dense feature maps from specified layers.

    For each layer, selects the top-K channels by L2
    magnitude and renders them as JET heatmaps.
    Produces:
      - Per-channel images  {stem}_{layer_name}_ch{i}.jpg
      - Per-layer grid      {stem}_{layer_name}_grid.jpg
      - All-layer summary   {stem}_features.jpg
    """

    # --------------------------------------------------------
    # Determine layers to visualize
    # --------------------------------------------------------

    default_layers = [
        "backbone.stages.1",
        "backbone.stages.2",
        "backbone.stages.3",
        "backbone.stages.4",
        "neck.fusion.cv",
    ]

    if args.layers:
        layer_names = [
            s.strip()
            for s in args.layers.split(",")
        ]
    else:
        layer_names = default_layers

    # --------------------------------------------------------
    # Hook to capture activations
    # --------------------------------------------------------

    root = model
    modules = dict(root.named_modules())

    captures = {}
    handles = []

    for name in layer_names:
        if name not in modules:
            print(
                f"[WARNING] Layer '{name}' not found, "
                "skipping."
            )
            continue

        def make_hook(n):
            def h(module, inp_t, out):
                if torch.is_tensor(out):
                    captures[n] = out.detach()
                elif isinstance(out, (list, tuple)):
                    captures[n] = out[0].detach()
            return h

        handles.append(
            modules[name].register_forward_hook(
                make_hook(name)
            )
        )

    if not handles:
        print("[ERROR] No valid layers found.")
        return

    # --------------------------------------------------------
    # Forward (no grad needed)
    # --------------------------------------------------------

    with torch.no_grad():
        features = model.backbone(inp)
        neck_out = model.neck(features)
        _ = model.detect(neck_out)

    # --------------------------------------------------------
    # Render
    # --------------------------------------------------------

    stem = Path(args.source).stem
    k = args.top_channels
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_grids = []

    for name in layer_names:
        if name not in captures:
            continue

        feat = captures[name]
        if feat.ndim != 4:
            print(
                f"[WARNING] {name}: expected [B,C,H,W], "
                f"got {feat.shape}. Skipping."
            )
            continue

        b, c, h, w = feat.shape
        feat = feat[0]  # [C, H, W]

        # Top-K channels by L2 magnitude
        channel_l2 = feat.pow(2).sum(
            dim=(1, 2)
        )
        topk_vals, topk_idx = channel_l2.topk(
            min(k, c)
        )

        # Build grid: up to 8 cols
        cell_h, cell_w = 128, 128
        n_show = len(topk_idx)
        cols = min(8, n_show)
        rows = (n_show + cols - 1) // cols

        grid = np.full(
            (rows * cell_h, cols * cell_w, 3),
            30,
            dtype=np.uint8
        )

        for i, ci in enumerate(topk_idx.tolist()):
            ch_map = feat[ci].cpu().numpy()
            ch_min = ch_map.min()
            ch_max = ch_map.max()
            if ch_max > ch_min:
                ch_map = (ch_map - ch_min) / (
                    ch_max - ch_min
                )
            else:
                ch_map = np.zeros_like(ch_map)

            ch_uint8 = np.uint8(ch_map * 255)
            heatmap = cv2.applyColorMap(
                ch_uint8,
                cv2.COLORMAP_JET
            )
            heatmap = cv2.resize(
                heatmap,
                (cell_w, cell_h),
                interpolation=cv2.INTER_LINEAR
            )

            r = i // cols
            col = i % cols
            y0 = r * cell_h
            x0 = col * cell_w
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = (
                heatmap
            )

            # Channel label
            cv2.putText(
                grid,
                f"ch{ci}",
                (x0 + 4, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # Title bar
        title_bar = np.full(
            (30, grid.shape[1], 3),
            0,
            dtype=np.uint8,
        )
        short_name = name.split(".")[-1]
        info = (
            f"{name}  "
            f"[1, {c}, {h}, {w}]  "
            f"top-{n_show} by L2"
        )
        cv2.putText(
            title_bar,
            info,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        layer_grid = np.vstack([title_bar, grid])

        # Save per-layer grid
        safe_name = name.replace(".", "_")
        grid_path = (
            out_dir / f"{stem}_{safe_name}_grid.jpg"
        )
        cv2.imwrite(str(grid_path), layer_grid)
        print(
            f"  {name:30s}  "
            f"grid → {grid_path.name}  "
            f"({n_show} channels, {h}x{w})"
        )

        all_grids.append(layer_grid)

    # --------------------------------------------------------
    # Summary: all layers stacked vertically
    # --------------------------------------------------------

    if all_grids:
        # Resize all grids to same width
        max_w = max(g.shape[1] for g in all_grids)
        resized = []
        for g in all_grids:
            if g.shape[1] < max_w:
                pad = np.full(
                    (g.shape[0], max_w - g.shape[1], 3),
                    30,
                    dtype=np.uint8,
                )
                g = np.hstack([g, pad])
            resized.append(g)

        summary = np.vstack(resized)
        summary_path = (
            out_dir / f"{stem}_features.jpg"
        )
        cv2.imwrite(str(summary_path), summary)
        print(
            f"\n  Summary: {summary_path}"
        )

    for h in handles:
        h.remove()

    print(
        f"\n  Done. {len(all_grids)} layers "
        f"visualized."
    )


# ============================================================
# FIND DETECTION SCORE
# ============================================================

def get_target_score(
    output,
    class_id=None,
    conf=0.25
):

    pred = extract_detection_tensor(
        output
    )

    if pred is None:

        raise RuntimeError(
            "Cannot find detection tensor."
        )

    print(
        "[INFO] Raw detection shape:",
        tuple(pred.shape)
    )

    # --------------------------------------------------------
    # Expected:
    #
    # [B, 84, 8500]
    #
    # Convert to:
    #
    # [B, 8500, 84]
    # --------------------------------------------------------

    if pred.shape[1] < pred.shape[2]:

        pred = pred.transpose(
            1,
            2
        )

    # --------------------------------------------------------
    # Classification part
    # --------------------------------------------------------

    cls_scores = pred[..., 4:]

    # --------------------------------------------------------
    # Specific class
    # --------------------------------------------------------

    if class_id is not None:

        scores = cls_scores[
            0,
            :,
            class_id
        ]

        # Remove extremely weak detections
        valid = scores > conf

        if valid.any():

            scores = scores[valid]

        score = scores.max()

        index = scores.argmax()

        print(
            f"[INFO] Target class = {class_id}"
        )

        print(
            f"[INFO] Target score = "
            f"{score.item():.6f}"
        )

        return score

    # --------------------------------------------------------
    # Automatically choose strongest detection
    # --------------------------------------------------------

    scores, classes = cls_scores[
        0
    ].max(
        dim=-1
    )

    valid = scores > conf

    if valid.any():

        scores_valid = scores[valid]
        classes_valid = classes[valid]

        best_index = scores_valid.argmax()

        score = scores_valid[
            best_index
        ]

        selected_class = classes_valid[
            best_index
        ].item()

    else:

        score, index = scores.max(
            dim=0
        )

        selected_class = classes[
            index
        ].item()

    print(
        f"[INFO] Auto target class = "
        f"{selected_class}"
    )

    print(
        f"[INFO] Target score = "
        f"{score.item():.6f}"
    )

    return score


# ============================================================
# GENERATE GRAD-CAM
# ============================================================

def generate_gradcam(
    activation,
    gradient
):

    if activation is None:

        raise RuntimeError(
            "Activation was not captured."
        )

    if gradient is None:

        raise RuntimeError(
            "Gradient was not captured."
        )

    print(
        "[INFO] Activation shape:",
        tuple(activation.shape)
    )

    print(
        "[INFO] Gradient shape:",
        tuple(gradient.shape)
    )

    if activation.ndim != 4:

        raise RuntimeError(
            "Grad-CAM requires [B,C,H,W]. "
            f"Got {activation.shape}"
        )

    # --------------------------------------------------------
    # Global average pooling
    # --------------------------------------------------------

    weights = gradient.mean(
        dim=(2, 3),
        keepdim=True
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = (
        weights * activation
    ).sum(
        dim=1,
        keepdim=True
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = F.relu(
        cam
    )

    cam = cam[0, 0]

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam = cam.detach()

    cam -= cam.min()

    max_value = cam.max()

    if max_value > 0:

        cam /= max_value

    return cam.cpu().numpy()


# ============================================================
# VISUALIZATION
# ============================================================

def make_heatmap(
    cam,
    image
):

    h, w = image.shape[:2]

    cam = cv2.resize(
        cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    cam_uint8 = np.uint8(
        cam * 255
    )

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    return heatmap


def make_overlay(
    image,
    heatmap,
    alpha
):

    return cv2.addWeighted(
        image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )


def add_title(
    image,
    title
):

    output = image.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 45),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        title,
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output


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
    # Device
    # --------------------------------------------------------

    if args.device.startswith("cuda"):

        if not torch.cuda.is_available():

            print(
                "[WARNING] CUDA unavailable. "
                "Using CPU."
            )

            device = torch.device("cpu")

        else:

            device = torch.device(
                args.device
            )

    else:

        device = torch.device("cpu")

    print(
        "[INFO] Device:",
        device
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print(
        "\n[INFO] Loading LIGHTEDGEDET..."
    )

    yolo = YOLO(
        args.weights
    )

    model = yolo.model

    model = model.to(
        device
    )

    # IMPORTANT:
    #
    # Do NOT use torch.no_grad()
    #
    # YOLO() freezes all parameters on load.
    # We need gradients to flow for Grad-CAM.
    #
    model.requires_grad_(True)
    model.eval()

    # --------------------------------------------------------
    # Load image (shared)
    # --------------------------------------------------------

    image = cv2.imread(
        args.source
    )

    if image is None:

        raise FileNotFoundError(
            args.source
        )

    # --------------------------------------------------------
    # Features mode — early exit
    # --------------------------------------------------------

    if args.mode == "features":

        input_tensor = preprocess(
            image,
            args.imgsz,
            device
        )

        visualize_features(
            model,
            input_tensor,
            args
        )

        return

    # --------------------------------------------------------
    # Grad-CAM mode
    # --------------------------------------------------------

    model.requires_grad_(True)

    # --------------------------------------------------------
    # Get target module
    # --------------------------------------------------------

    target_module = get_target_module(
        yolo,
        args.target
    )

    hook = FeatureHook(
        target_module,
        args.target
    )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    original = image.copy()

    input_tensor = preprocess(
        image,
        args.imgsz,
        device
    )

    input_tensor.requires_grad_(
        True
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    print(
        "\n[INFO] Forward..."
    )

    model.zero_grad(
        set_to_none=True
    )

    # Route through backbone → neck → forward_head
    # (raw class logits).  The default decode path
    # (DFL+bbox) kills gradients for high-conf
    # predictions.

    features = model.backbone(input_tensor)
    neck_out = model.neck(features)

    preds = model.detect.forward_head(
        neck_out,
        model.detect.cv2,
        model.detect.cv3
    )

    # --------------------------------------------------------
    # Target — top-k class-specific mean for dense gradients.
    # "global max" produces sparse gradients because only
    # one (class, anchor) pair gets a nonzero gradient.
    # top-k mean aggregates over many anchors, giving
    # denser, more spatially meaningful CAM.
    # ----------------------------------------------------------------

    cls_scores = preds["scores"]  # [B, nc, N]

    print(
        "[INFO] scores shape:",
        tuple(cls_scores.shape)
    )

    if args.class_id is not None:

        target_cls = args.class_id

    else:

        # Auto-detect: pick the class with highest max logit
        per_class_max = cls_scores[0].max(dim=1).values
        target_cls = int(per_class_max.argmax().item())

    class_logits = cls_scores[0, target_cls]  # [N]
    target_score = class_logits.clamp(min=0).sum()

    print(
        f"[INFO] Target class = {target_cls}"
    )
    print(
        f"[INFO] Target score = "
        f"{target_score.item():.6f}"
    )

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    print(
        "[INFO] Backward..."
    )

    target_score.backward()

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam = generate_gradcam(
        hook.activation,
        hook.activation.grad
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    heatmap = make_heatmap(
        cam,
        original
    )

    overlay = make_overlay(
        original,
        heatmap,
        args.alpha
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    stem = Path(
        args.source
    ).stem

    prefix = os.path.join(
        args.output,
        f"{stem}_{args.target}"
    )

    original_out = add_title(
        original,
        "Original"
    )

    heatmap_out = add_title(
        heatmap,
        f"Grad-CAM: {args.target.upper()}"
    )

    overlay_out = add_title(
        overlay,
        f"LIGHTEDGEDET - {args.target.upper()}"
    )

    # --------------------------------------------------------
    # Side-by-side
    # --------------------------------------------------------

    comparison = np.concatenate(
        [
            original_out,
            heatmap_out,
            overlay_out
        ],
        axis=1
    )

    cv2.imwrite(
        prefix + "_original.jpg",
        original
    )

    cv2.imwrite(
        prefix + "_heatmap.jpg",
        heatmap
    )

    cv2.imwrite(
        prefix + "_overlay.jpg",
        overlay
    )

    cv2.imwrite(
        prefix + "_comparison.jpg",
        comparison
    )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    hook.close()

    print(
        "\n" + "=" * 70
    )

    print(
        "DONE"
    )

    print(
        "=" * 70
    )

    print(
        "Target:",
        args.target
    )

    print(
        "Output:",
        prefix + "_comparison.jpg"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()