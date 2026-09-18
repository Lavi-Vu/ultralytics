"""Export a LightEdgeDet checkpoint to a raw-output Hailo-8L HEF.

The output has eight tensors (box and class logits for strides 8/16/32/64).
Decode boxes and run NMS on the host; this is not a standard YOLOv8 NMS HEF.
Level 0 is the default for memory-limited hosts; level 2 needs substantially more RAM.

Example:
    python scripts/export_lightedgedet_hailo.py runs/train/weights/best.pt /path/to/calibration/images \
        --output-dir /path/to/export
"""

from __future__ import annotations

import argparse
import importlib.metadata
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx
    from hailo_sdk_client import ClientRunner


def end_nodes(model: onnx.ModelProto, levels: int) -> list[str]:
    """Find the paired final box and class convolutions in export order."""
    names = {node.name for node in model.graph.node}
    ends = [f"/detect/cv{branch}.{scale}/cv{branch}.{scale}.2/Conv" for scale in range(levels) for branch in (2, 3)]
    missing = set(ends) - names
    if missing:
        raise ValueError(f"missing detection outputs in ONNX graph: {sorted(missing)}")
    return ends


def parse_checkpoint(checkpoint: Path, output_dir: Path, imgsz: int) -> tuple[Path, Path]:
    """Export ONNX and parse its eight raw heads into a Hailo archive."""
    import onnx
    from hailo_sdk_client import ClientRunner
    from hailo_sdk_client.model_translator.fuser.fuser import HailoNNFuser

    from ultralytics import YOLO

    local_checkpoint = output_dir / checkpoint.name
    if checkpoint.resolve() != local_checkpoint.resolve():
        shutil.copy2(checkpoint, local_checkpoint)
    model = YOLO(str(local_checkpoint))
    head = model.model.model[-1]
    if model.task != "lightedgedet" or head.reg_max != 1 or head.nl != 4:
        raise ValueError("this export path requires a four-level LightEdgeDet checkpoint with reg_max=1")
    onnx_path = Path(model.export(format="onnx", imgsz=imgsz, opset=14, simplify=False, device="cpu"))
    graph = onnx.load(str(onnx_path))
    ends = end_nodes(graph, head.nl)

    # DFC 3.34.0 crashes while folding the SE/global-pool Conv1x1 chain. Skipping
    # only that optional fuser pass preserves the graph; verify_native checks it.
    original = HailoNNFuser._handle_conv1x1_after_global_avgpool
    try:
        HailoNNFuser._handle_conv1x1_after_global_avgpool = lambda self: None
        runner = ClientRunner(hw_arch="hailo8l")
        runner.translate_onnx_model(str(onnx_path), "lightedgedet", end_node_names=ends)
    finally:
        HailoNNFuser._handle_conv1x1_after_global_avgpool = original
    har_path = output_dir / "lightedgedet_raw.har"
    runner.save_har(str(har_path))
    verify_native(graph, ends, runner, imgsz)
    return har_path, onnx_path


def verify_native(graph: onnx.ModelProto, ends: list[str], runner: ClientRunner, imgsz: int) -> None:
    """Check that parsed floating-point heads match the ONNX export."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from hailo_sdk_client import InferenceContext

    outputs = {node.name: node.output[0] for node in graph.graph.node}
    for name in ends:
        graph.graph.output.append(onnx.helper.make_tensor_value_info(outputs[name], onnx.TensorProto.FLOAT, None))
    sample = np.random.default_rng(0).random((1, 3, imgsz, imgsz), dtype=np.float32)
    expected = ort.InferenceSession(graph.SerializeToString(), providers=["CPUExecutionProvider"]).run(
        None, {graph.graph.input[0].name: sample}
    )[1:]
    with runner.infer_context(InferenceContext.SDK_NATIVE) as context:
        actual = runner.infer(context, sample.transpose(0, 2, 3, 1))
    errors = [np.max(np.abs(ref.transpose(0, 2, 3, 1) - got)) for ref, got in zip(expected, actual)]
    if len(errors) != len(ends) or max(errors) > 0.01:
        raise RuntimeError(f"Hailo parser changed the detection outputs: max errors {errors}")
    print(f"Native Hailo graph verified; maximum absolute error {max(errors):.6g}", flush=True)


def calibration_dataset(directory: Path, imgsz: int, limit: int):
    """Yield RGB letterboxed calibration images in the exporter's 0–255 range."""
    import cv2
    import numpy as np
    import tensorflow as tf

    from ultralytics.data.augment import LetterBox

    paths = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})[:limit]
    if not paths:
        raise ValueError(f"no calibration images in {directory}")
    letterbox = LetterBox(new_shape=(imgsz, imgsz))

    def images():
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"cannot read calibration image {path}")
            yield cv2.cvtColor(letterbox(image=image), cv2.COLOR_BGR2RGB).astype(np.float32), {}

    dataset = tf.data.Dataset.from_generator(
        images, output_signature=(tf.TensorSpec(shape=(imgsz, imgsz, 3), dtype=tf.float32), {})
    )
    return dataset.apply(tf.data.experimental.assert_cardinality(len(paths))), len(paths)


def main() -> None:
    """Build, calibrate, and compile a Hailo-8L model."""
    import yaml
    from hailo_sdk_client import ClientRunner

    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("calibration_images", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--optimization-level", type=int, default=0, choices=(0, 1, 2))
    args = parser.parse_args()
    if importlib.metadata.version("hailo-dataflow-compiler") != "3.34.0":
        parser.error("this LightEdgeDet parser workaround was validated only with Hailo DFC 3.34.0")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hef_path = args.output_dir / "lightedgedet_hailo8l_raw.hef"
    if hef_path.exists():
        parser.error(f"refusing to overwrite {hef_path}")
    har_path, onnx_path = parse_checkpoint(args.checkpoint, args.output_dir, args.imgsz)
    dataset, count = calibration_dataset(args.calibration_images, args.imgsz, args.limit)
    runner = ClientRunner(har=str(har_path))
    script = [
        "input_normalization = normalization([0, 0, 0], [255, 255, 255])",
        f"model_optimization_config(calibration, calibset_size={count})",
        "pre_quantization_optimization(global_avgpool_reduction, layers=avgpool1, division_factors=[4, 4])",
        f"model_optimization_flavor(optimization_level={args.optimization_level})",
    ]
    if args.optimization_level == 2:
        script.append(f"post_quantization_optimization(finetune, policy=enabled, dataset_size={count})")
    runner.load_model_script("\n".join(script))
    runner.optimize(dataset)
    runner.save_har(str(args.output_dir / "lightedgedet_optimized.har"))
    hef_path.write_bytes(runner.compile())
    (args.output_dir / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "checkpoint": str(args.checkpoint.resolve()),
                "onnx": str(onnx_path),
                "hef": str(hef_path),
                "hailo_arch": "hailo8l",
                "imgsz": args.imgsz,
                "calibration_images": count,
                "optimization_level": args.optimization_level,
                "output_type": "raw_box_and_class_logits",
                "strides": [8, 16, 32, 64],
                "reg_max": 1,
                "names": YOLO(str(args.checkpoint)).names,
                "nms": False,
            },
            sort_keys=False,
        )
    )
    print(f"Hailo-8L HEF: {hef_path}")


if __name__ == "__main__":
    main()
