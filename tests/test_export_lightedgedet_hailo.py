"""Tests for LightEdgeDet's Hailo output selection."""

from types import SimpleNamespace

import pytest

from scripts.export_lightedgedet_hailo import end_nodes


def graph_with(names):
    """Build a minimal graph-like object for output selection."""
    return SimpleNamespace(graph=SimpleNamespace(node=[SimpleNamespace(name=name) for name in names]))


def test_end_nodes_selects_box_class_pairs_in_stride_order():
    """The Hailo outputs must remain paired for host-side decoding."""
    expected = [f"/detect/cv{branch}.{scale}/cv{branch}.{scale}.2/Conv" for scale in range(4) for branch in (2, 3)]
    assert end_nodes(graph_with(reversed(expected)), 4) == expected


def test_end_nodes_rejects_incompatible_head():
    """A different ONNX head must not produce a plausible but incorrect HEF."""
    with pytest.raises(ValueError, match="missing detection outputs"):
        end_nodes(graph_with([]), 4)
