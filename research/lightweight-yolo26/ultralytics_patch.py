"""Monkey-patch Ultralytics' parse_model to support custom research module names in YAML configs.

Adds ReparamC3k2, GatedC3k2, and GhostC3k2 as first-class modules
in base_modules and repeat_modules frozensets, so they get proper
width/depth scaling in YAML configs like standard C3k2.

Usage:
    import ultralytics_patch          # auto-applies on import
    from ultralytics import YOLO
    model = YOLO("cfg/yolo26n_reparam.yaml")
"""

import sys
from pathlib import Path

import torch.nn as nn

# Ensure research dir is on path (at END, not front) for custom module imports
_research_dir = str(Path(__file__).resolve().parent)
if _research_dir not in sys.path:
    sys.path.append(_research_dir)

import ultralytics.nn.tasks as tasks
from ultralytics.nn.modules.block import C2f, C3k2
from models.modules.reparam_block import ReparamBottleneck
from models.modules.channel_gate import ChannelGate
from models.modules.ghost_c3k2 import GhostC3k2


# ---------------------------------------------------------------------------
# Custom module class definitions — same __init__ signature as C3k2 so that
# YAML args (c2, c3k, e, ...) map correctly after width/depth scaling.
# ---------------------------------------------------------------------------

class ReparamC3k2(C2f):
    """C3k2 variant using ReparamBottleneck (3×3 + 1×1 reparam convs)."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            ReparamBottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class GatedC3k2(nn.Module):
    """Wraps a standard C3k2 with learnable channel gating on its input."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, attn=False, g=1, shortcut=True):
        super().__init__()
        self.gate = ChannelGate(c1, init_keep_prob=0.85, regularize=True)
        self.c3k2 = C3k2(c1, c2, n, c3k, e, attn, g, shortcut)

    def forward(self, x):
        return self.c3k2(self.gate(x))

    @property
    def m(self):
        return self.c3k2.m

    @property
    def cv1(self):
        return self.c3k2.cv1

    @property
    def cv2(self):
        return self.c3k2.cv2


# Register all custom classes in the tasks module so parse_model can resolve
# them via globals()["ReparamC3k2"] etc.
for _cls in [ReparamC3k2, GatedC3k2, GhostC3k2]:
    setattr(tasks, _cls.__name__, _cls)


# ---------------------------------------------------------------------------
# Monkey-patch parse_model to include our classes in base_modules and
# repeat_modules frozensets.
# ---------------------------------------------------------------------------

import inspect
import textwrap


def _patch_parse_model():
    """Replace tasks.parse_model with a version that includes our custom modules
    in the base_modules and repeat_modules frozensets.
    """
    source = textwrap.dedent(inspect.getsource(tasks.parse_model))

    lines = source.split("\n")
    seen_base_modules = False
    seen_repeat_modules = False
    patched_base = False
    patched_repeat = False
    out = []

    for line in lines:
        if "base_modules = frozenset(" in line:
            seen_base_modules = True
        if "repeat_modules = frozenset(" in line:
            seen_repeat_modules = True
            seen_base_modules = False

        # Patch within base_modules block
        if seen_base_modules and not patched_base and "C3k2," in line:
            line = line.replace("C3k2,", "C3k2, ReparamC3k2, GatedC3k2, GhostC3k2,")
            patched_base = True
        # Patch within repeat_modules block
        if seen_repeat_modules and not patched_repeat and "C3k2," in line:
            line = line.replace("C3k2,", "C3k2, ReparamC3k2, GatedC3k2, GhostC3k2,")
            patched_repeat = True

        out.append(line)

    modified = "\n".join(out)

    # Compile in the tasks module's namespace so class references resolve
    local_ns = {}
    exec(compile(modified, "<patched_parse_model>", "exec"), tasks.__dict__, local_ns)
    tasks.parse_model = local_ns["parse_model"]


_patch_parse_model()