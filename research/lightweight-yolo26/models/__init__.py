from .yolo26_n_baseline import YOLO26NBaseline
from .yolo26_n_reparam import YOLO26NReparam
from .yolo26_n_gate import YOLO26NGate
from .yolo26_n_combined import YOLO26NCombined
from .yolo26_neck_ghost import YOLO26NeckGhost
from .yolo26_neck_light import YOLO26NeckLight
from .yolo26_neck_wide import YOLO26NeckWide

MODEL_REGISTRY = {
    "baseline": YOLO26NBaseline,
    "reparam": YOLO26NReparam,
    "gate": YOLO26NGate,
    "combined": YOLO26NCombined,
    "neck-ghost": YOLO26NeckGhost,
    "neck-light": YOLO26NeckLight,
    "neck-wide": YOLO26NeckWide,
}