"""Tests for LightEdgeDet's Hailo output selection."""

from types import SimpleNamespace

import numpy as np
import pytest

from ultralytics.engine.exporter import _hailo_lightedgedet_output_names


def test_output_names_orders_box_streams_before_class_streams():
    """Hailo streams must stay paired for host-side LightEdgeDet decoding."""
    hn = {"name": "test", "layers": {f"test/output_layer{i}": {"input": [f"test/conv{i}"]} for i in range(1, 9)}}
    assert _hailo_lightedgedet_output_names(hn, 8) == [
        "test/conv1",
        "test/conv3",
        "test/conv5",
        "test/conv7",
        "test/conv2",
        "test/conv4",
        "test/conv6",
        "test/conv8",
    ]


def test_quantized_hailo_outputs_dequantize_without_layout_changes():
    """Raw Hailo streams are dequantized on the host after the smaller PCIe transfer."""
    from ultralytics.nn.backends.hailo import HailoBackend

    output = np.array([[[[0, 8], [16, 255]]]], dtype=np.uint8)
    actual = HailoBackend._dequantize_output(output, (0.25, 8))
    np.testing.assert_array_equal(actual, np.array([[[[-2, 0], [2, 61.75]]]], dtype=np.float32))
    assert actual.shape == output.shape


def test_fuse_removes_lightedgedet_batch_norm():
    """LightEdgeDet's custom backbone and neck must fuse their raw Conv-BatchNorm sequences."""
    import torch

    from ultralytics import YOLO

    model = YOLO("ultralytics/cfg/models/lightedgedet_nano.yaml").model
    model.fuse(verbose=False)
    assert not any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.modules())


@pytest.mark.parametrize("classes", [4, 80])
@pytest.mark.parametrize("batch", [1, 2])
def test_raw_forward_matches_detect_with_shuffled_streams(classes, batch):
    """Named streams preserve roles even with identical channel counts and arbitrary runtime order."""
    import torch

    from ultralytics.nn.backends.hailo import HailoBackend
    from ultralytics.nn.modules.head import Detect

    torch.manual_seed(0)
    head = Detect(nc=classes, reg_max=1, end2end=False, ch=(16, 16, 16, 16)).eval()
    head.stride = torch.tensor([8, 16, 32, 64])
    boxes = [torch.rand(batch, 4, size, size) for size in (8, 4, 2, 1)]
    scores = [torch.randn(batch, classes, size, size) for size in (8, 4, 2, 1)]
    paired = [value for pair in zip(boxes, scores) for value in pair]
    hn = {"name": "test", "layers": {f"test/output_layer{i}": {"input": [f"test/conv{i}"]} for i in range(1, 9)}}
    results = {f"test/conv{i}": value.permute(0, 2, 3, 1).numpy() for i, value in enumerate(paired, 1)}
    backend = HailoBackend.__new__(HailoBackend)
    backend.metadata = {"output_names": _hailo_lightedgedet_output_names(hn, 8)}
    backend.task, backend.end2end, backend._anchors = "detect", False, None
    backend.output_quantized, backend.output_quant_params = False, {}
    backend.input_info = SimpleNamespace(name="input", shape=(64, 64, 3))
    backend.output_infos = [SimpleNamespace(name=name) for name in reversed(results)]
    backend.model = SimpleNamespace(infer=lambda inputs: results)
    actual = backend.forward(torch.zeros(batch, 3, 64, 64))
    expected = head._inference(
        {
            "boxes": torch.cat([x.flatten(2) for x in boxes], 2),
            "scores": torch.cat([x.flatten(2) for x in scores], 2),
            "feats": boxes,
        }
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(backend.forward(torch.zeros(batch, 3, 64, 64)), expected)

    # The existing YOLO26 branch still returns top-k xyxy/conf/class detections.
    backend.end2end = True
    decoded = backend._decode_raw([results[name] for name in _hailo_lightedgedet_output_names(hn, 8)])
    assert isinstance(decoded, np.ndarray)
    assert decoded.shape == (batch, 300, 6)
    assert np.all((decoded[..., 4] >= 0) & (decoded[..., 4] <= 1))


@pytest.mark.parametrize("raw_lightedgedet", [False, True])
def test_load_selects_host_nms_only_for_lightedgedet(tmp_path, monkeypatch, raw_lightedgedet):
    """Metadata loading must not disable legacy YOLO26 top-k postprocessing."""
    import sys
    from contextlib import nullcontext

    from ultralytics.nn.backends.hailo import HailoBackend
    from ultralytics.utils import YAML

    group = SimpleNamespace(activate=lambda _: nullcontext(), create_params=lambda: None)
    device = SimpleNamespace(configure=lambda *_: [group])
    hef = SimpleNamespace(
        get_input_vstream_infos=lambda: [SimpleNamespace(shape=(640, 640, 3))],
        get_output_vstream_infos=list,
    )
    params = SimpleNamespace(make=lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "hailo_platform",
        SimpleNamespace(
            HEF=lambda _: hef,
            ConfigureParams=SimpleNamespace(create_from_hef=lambda *args, **kwargs: None),
            FormatType=SimpleNamespace(UINT8=0, FLOAT32=1),
            HailoStreamInterface=SimpleNamespace(PCIe=0),
            InferVStreams=lambda *args: nullcontext(SimpleNamespace()),
            InputVStreamParams=params,
            OutputVStreamParams=params,
            VDevice=lambda: nullcontext(device),
        ),
    )
    (tmp_path / "model.hef").touch()
    metadata = {"task": "lightedgedet" if raw_lightedgedet else "detect", "nms": False}
    if raw_lightedgedet:
        metadata.update(output_type="raw_box_and_class_logits", end2end=False)
    YAML.save(tmp_path / "metadata.yaml", metadata)
    backend = HailoBackend(tmp_path, device="cpu")
    assert backend.end2end is not raw_lightedgedet
