#!/usr/bin/env python3
# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Compile a LightEdgeDet checkpoint to Hailo HEF without calling ``YOLO.export()``.

Example:
    python scripts/export_lightedgedet_hailo.py train-10/weights/best_fused.pt \
        --data ultralytics/cfg/datasets/coco.yaml --imgsz 512 --calib-size 1024
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def parse_args() -> argparse.Namespace:
    """Parse standalone Hailo export options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="LightEdgeDet .pt checkpoint")
    parser.add_argument(
        "--data", type=Path, required=True, help="Dataset YAML used for representative calibration images"
    )
    parser.add_argument(
        "--dataset-root", type=Path, help="Resolved dataset root; required when YAML uses a relative path"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square Hailo input resolution")
    parser.add_argument("--calib-size", type=int, default=1024, help="Number of representative calibration images")
    parser.add_argument("--arch", default="hailo8l", choices=("hailo8", "hailo8l"), help="Target Hailo architecture")
    parser.add_argument("--output", type=Path, help="Output directory (default: <weights>_hailo_model)")
    parser.add_argument("--bias-correction", action="store_true", help="Enable slower Hailo bias correction")
    parser.add_argument("--finetune", action="store_true", help="Enable resource-intensive Hailo QAT")
    parser.add_argument("--finetune-size", type=int, default=128, help="Representative images used by Hailo QAT")
    parser.add_argument("--finetune-batch", type=int, default=1, help="Hailo QAT batch size; 1 minimizes memory use")
    return parser.parse_args()


def dataset_images(data_file: Path, calibration_size: int, dataset_root: Path | None = None) -> list[Path]:
    """Return evenly distributed validation images resolved from an Ultralytics dataset YAML."""
    data = yaml.safe_load(data_file.read_text())
    yaml_root = Path(data.get("path", ""))
    candidates = [dataset_root] if dataset_root else []
    if yaml_root.is_absolute():
        candidates.append(yaml_root)
    else:
        candidates.extend(
            (
                data_file.parent / yaml_root,
                Path.cwd().parent / "datasets" / yaml_root,
                Path.home() / "datasets" / yaml_root,
            )
        )
    root = next((path.resolve() for path in candidates if path and path.is_dir()), None)
    if root is None:
        tried = ", ".join(str(path) for path in candidates if path)
        raise FileNotFoundError(f"Dataset root does not exist. Pass --dataset-root. Tried: {tried}")
    source = data.get("val") or data.get("train")
    if not source:
        raise ValueError(f"Dataset YAML has neither 'val' nor 'train': {data_file}")
    sources = source if isinstance(source, list) else [source]
    images = []
    for item in sources:
        path = Path(item)
        path = path if path.is_absolute() else root / path
        if path.is_file() and path.suffix.lower() == ".txt":
            images.extend(
                (Path(x.strip()) if Path(x.strip()).is_absolute() else path.parent / x.strip())
                for x in path.read_text().splitlines()
                if x.strip()
            )
        elif path.is_dir():
            images.extend(p for p in path.rglob("*") if p.suffix.lower() in {".bmp", ".jpeg", ".jpg", ".png"})
    images = sorted(p.resolve() for p in images if p.is_file())
    if not images:
        raise FileNotFoundError(f"No images found from dataset YAML: {data_file}")
    indices = np.linspace(0, len(images) - 1, min(calibration_size, len(images)), dtype=int)
    return [images[i] for i in indices]


def letterbox_rgb(path: Path, size: int) -> np.ndarray:
    """Read one image as RGB HWC float32 in the 0-255 range expected by Hailo normalization."""
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Unable to read calibration image: {path}")
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    resized = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = size - resized.shape[1], size - resized.shape[0]
    left, right = round(pad_w / 2 - 0.1), round(pad_w / 2 + 0.1)
    top, bottom = round(pad_h / 2 - 0.1), round(pad_h / 2 + 0.1)
    image = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(image[..., ::-1], dtype=np.float32)


def export_onnx(model: torch.nn.Module, output: Path, size: int, metadata: dict) -> None:
    """Export the model graph directly to ONNX at opset 14 for attention compatibility."""
    image = torch.zeros(1, 3, size, size)
    kwargs = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    torch.onnx.export(
        model,
        image,
        str(output),
        opset_version=14,
        input_names=["images"],
        output_names=["output0"],
        dynamic_axes=None,
        do_constant_folding=True,
        **kwargs,
    )
    import onnx

    onnx_model = onnx.load(output)
    for key, value in metadata.items():
        item = onnx_model.metadata_props.add()
        item.key, item.value = key, str(value)
    onnx.save(onnx_model, output)


def main() -> None:
    """Export the checkpoint to ONNX and compile it to a Hailo HEF."""
    args = parse_args()
    try:
        import tensorflow as tf
        from hailo_sdk_client import ClientRunner
        from hailo_sdk_client.model_translator.fuser.fuser import HailoNNFuser

        from ultralytics import YOLO
        from ultralytics.engine.exporter import _hailo_lightedgedet_fusion_layers
    except ImportError as e:
        raise SystemExit(
            "Install Ultralytics, TensorFlow, and the Hailo Dataflow Compiler before running this script."
        ) from e

    yolo = YOLO(args.weights)
    if yolo.task != "lightedgedet":
        raise ValueError(f"Expected a LightEdgeDet checkpoint, received task='{yolo.task}'.")
    model = yolo.model.fuse().eval()
    head = model.model[-1]
    if getattr(head, "reg_max", None) != 1 or len(head.cv2) != 4:
        raise ValueError("Hailo export requires a four-level LightEdgeDet Detect head with reg_max=1.")
    head.export, head.format = True, "onnx"

    output_dir = args.output or args.weights.with_suffix("").with_name(f"{args.weights.stem}_hailo_model")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    onnx_file = output_dir / f"{args.weights.stem}.onnx"
    metadata = {
        "task": "lightedgedet",
        "imgsz": [args.imgsz, args.imgsz],
        "stride": int(max(model.stride)),
        "names": yolo.names,
    }
    export_onnx(model, onnx_file, args.imgsz, metadata)

    # Keep each scale's box/class outputs adjacent; class activations below select the odd entries.
    end_nodes = [f"/detect/cv{branch}.{level}/cv{branch}.{level}.2/Conv" for level in range(4) for branch in (2, 3)]
    runner = ClientRunner(hw_arch=args.arch)
    original_fuser = HailoNNFuser._handle_conv1x1_after_global_avgpool
    try:
        HailoNNFuser._handle_conv1x1_after_global_avgpool = lambda _: None
        runner.translate_onnx_model(str(onnx_file), args.weights.stem, end_node_names=end_nodes)
    finally:
        HailoNNFuser._handle_conv1x1_after_global_avgpool = original_fuser
    fusion_layers = _hailo_lightedgedet_fusion_layers(runner.get_hn_dict())

    images = dataset_images(args.data, args.calib_size, args.dataset_root)
    model_script = [
        "input_normalization = normalization([0, 0, 0], [255, 255, 255])",
        f"model_optimization_config(calibration, calibset_size={len(images)})",
        "model_optimization_config(checker_cfg, policy=disabled)",
        "pre_quantization_optimization(global_avgpool_reduction, layers=avgpool1, division_factors=[4, 4])",
        "model_optimization_flavor(optimization_level=2)",
        "performance_param(compiler_optimization_level=max)",
        f"quantization_param([{', '.join(fusion_layers)}], precision_mode=a16_w16)",
    ]
    if args.bias_correction:
        model_script.append("post_quantization_optimization(bias_correction, policy=enabled)")
    if args.finetune:
        model_script.append(
            f"post_quantization_optimization(finetune, policy=enabled, dataset_size={min(args.finetune_size, len(images))}, "
            f"batch_size={args.finetune_batch})"
        )
    else:
        model_script.append("post_quantization_optimization(finetune, policy=disabled)")
    runner.load_model_script("\n".join(model_script))

    def calibration_dataset():
        for image_path in images:
            yield letterbox_rgb(image_path, args.imgsz), {}

    runner.optimize(
        lambda: tf.data.Dataset.from_generator(
            calibration_dataset,
            output_signature=(tf.TensorSpec(shape=(args.imgsz, args.imgsz, 3), dtype=tf.float32), {}),
        )
    )
    hef_file = output_dir / f"{args.weights.stem}.hef"
    hef_file.write_bytes(runner.compile())
    metadata.update(
        nms=False,
        output_type="raw_box_and_class_logits",
        class_activation="raw",
        output_quantized=False,
    )
    (output_dir / "metadata.yaml").write_text(yaml.safe_dump(metadata, sort_keys=False))
    print(f"Hailo export complete: {output_dir}")


if __name__ == "__main__":
    main()
