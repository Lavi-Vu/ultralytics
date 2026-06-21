# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""EdgeRF-YOLO model definition for hybrid CNN-Transformer real-time detection."""

from pathlib import Path
import re

from ultralytics.nn.tasks import DetectionModel, yaml_model_load


def guess_edgerf_scale(model_path):
    """Extract the scale character (n, s, m, l, x) from EdgeRF model filenames."""
    try:
        return re.search(r"edgerf-([nslmx])", Path(model_path).stem).group(1)
    except AttributeError:
        return ""


class EdgeRFDetectionModel(DetectionModel):
    """EdgeRF-YOLO detection model with proper scale detection from filename."""

    def __init__(self, cfg="edgerf-n.yaml", ch=3, nc=None, verbose=True):
        if isinstance(cfg, dict):
            super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        else:
            cfg_path = Path(cfg)
            d = yaml_model_load(cfg_path)
            scale = guess_edgerf_scale(cfg_path)
            if scale and "scales" in d:
                d["scale"] = scale
            super().__init__(cfg=d, ch=ch, nc=nc, verbose=verbose)
