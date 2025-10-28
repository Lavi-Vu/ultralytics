from pathlib import Path
from ultralytics.utils.torch_utils import strip_optimizer

for f in Path("runs/detect/train34/weights").rglob("*.pt"):
   strip_optimizer(f)