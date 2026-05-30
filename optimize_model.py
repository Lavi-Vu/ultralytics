from pathlib import Path
from ultralytics.utils.torch_utils import strip_optimizer

for f in Path("/home/lavi/Documents/Lightweight-Transformer-YOLO-Hybrid-Models/train-23/weights/best.pt").rglob("*.pt"):
   strip_optimizer(f)