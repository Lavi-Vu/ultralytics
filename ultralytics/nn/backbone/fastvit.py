"""
fastvit_backbone.py
===================
FastViT (Apple ICCV 2023) backbone wrapper for Ultralytics YOLO26.

ROOT CAUSE OF THE CRASH
------------------------
Ultralytics' parse_model() tracks channel widths in a flat list `ch[]`.
When a layer outputs a single tensor, it appends one entry.
FastViTBackbone returns a **list** of tensors (one per FPN stage), so
parse_model can't infer any channels — it defaults to 0, and every
downstream Conv gets weight shape [out, 0, k, k] → crash.

FIX
---
We patch parse_model to:
  1. Detect FastViTBackbone entries.
  2. Append ONE ch[] entry per output stage (not just one total).
  3. Use a saved-layer index map so downstream Index / Conv layers
     find the right channel count.

INSTALL
-------
    pip install timm>=0.9.0

REGISTER — choose one method
-----------------------------
  A) Source edit (permanent, recommended):
       cp fastvit_backbone.py  ultralytics/nn/modules/fastvit_backbone.py
       # add to ultralytics/nn/modules/__init__.py:
       from .fastvit_backbone import FastViTBackbone  # noqa
       # Then call patch_parse_model() once at import time (see bottom of file)

  B) Runtime patch (no source edits):
       from fastvit_backbone import patch_ultralytics
       patch_ultralytics()          # must run BEFORE `from ultralytics import YOLO`
       from ultralytics import YOLO
       model = YOLO("yolo26-fastvit.yaml")

VARIANT CHANNEL TABLE
---------------------
  fastvit_t8    [48,  96, 192, 384]  76.2% top-1  0.8ms iP12
  fastvit_t12   [48,  96, 192, 384]  79.3%         1.2ms
  fastvit_s12   [64, 128, 256, 512]  79.9%         1.4ms
  fastvit_sa12  [64, 128, 256, 512]  80.9%         1.6ms  ← default
  fastvit_sa24  [64, 128, 256, 512]  82.7%         2.6ms
  fastvit_sa36  [64, 128, 256, 512]  83.6%         3.5ms
  fastvit_ma36  [76, 152, 304, 608]  83.9%         4.6ms
"""

from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

# Official output channels per stage [P2/4, P3/8, P4/16, P5/32]
FASTVIT_CHANNELS: dict[str, list[int]] = {
    "fastvit_t8":   [48,  96,  192, 384],
    "fastvit_t12":  [48,  96,  192, 384],
    "fastvit_s12":  [64,  128, 256, 512],
    "fastvit_sa12": [64,  128, 256, 512],
    "fastvit_sa24": [64,  128, 256, 512],
    "fastvit_sa36": [64,  128, 256, 512],
    "fastvit_ma36": [76,  152, 304, 608],
}


class FastViTBackbone(nn.Module):
    """
    FastViT multi-scale feature extractor for Ultralytics YOLO26.

    forward() returns a **list** of feature tensors, one per requested stage.
    The patched parse_model() handles this list and registers each stage's
    channel count into ch[] so downstream layers resolve sizes correctly.

    Args:
        model_name  (str):       timm model name.         Default: "fastvit_sa12"
        pretrained  (bool):      Load ImageNet weights.   Default: True
        out_indices (list[int]): Stages to return.        Default: [1,2,3]
                                  0=P2/4  1=P3/8  2=P4/16  3=P5/32
        drop_path   (float):     Stochastic depth.        Default: 0.1
    """

    def __init__(
        self,
        model_name:  str        = "fastvit_sa12",
        pretrained:  bool       = True,
        out_indices: list[int]  = (1, 2, 3),
        drop_path:   float      = 0.1,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "timm is required for FastViTBackbone.\n"
                "  pip install timm>=0.9.0"
            ) from exc

        self.model_name  = model_name
        self.out_indices = list(out_indices)

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=self.out_indices,
            drop_path_rate=drop_path,
        )

        # Resolve output channels for each stage
        if model_name in FASTVIT_CHANNELS:
            all_ch = FASTVIT_CHANNELS[model_name]
            self.out_channels: list[int] = [all_ch[i] for i in self.out_indices]
        else:
            dummy = torch.zeros(1, 3, 256, 256)
            with torch.no_grad():
                feats = self.model(dummy)
            self.out_channels = [f.shape[1] for f in feats]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Returns list of feature maps: [P3, P4, P5] for out_indices=[1,2,3]."""
        return self.model(x)

    def reparameterize(self) -> None:
        """Fuse RepMixer branches post-training for faster inference / export."""
        try:
            from timm.models import reparameterize_model
            self.model = reparameterize_model(self.model)
            print("[FastViTBackbone] ✓ Structural reparameterization complete.")
        except Exception:
            print("[FastViTBackbone] ⚠ reparameterize_model unavailable — skipping.")

    def extra_repr(self) -> str:
        return (f"model={self.model_name}, pretrained=True, "
                f"out_indices={self.out_indices}, out_channels={self.out_channels}")


# ─────────────────────────────────────────────────────────────────────────────
# parse_model patch
# ─────────────────────────────────────────────────────────────────────────────

def _build_patched_parse_model(original_parse_model):
    """
    Wraps the original parse_model, intercepting FastViTBackbone layers to
    register all their output channels into ch[] before continuing normally.

    Strategy:
      - Walk the layer list BEFORE the main loop.
      - For each FastViTBackbone entry at index i, record:
          _fastvit_channels[i] = [ch_stage0, ch_stage1, ...]
      - Inside the loop, when we hit FastViTBackbone:
          • Build the module.
          • Append each stage's channel to ch[] (not just one).
          • Set c2 to the last stage's channel (for the loop's final ch.append).
          • Use `continue` so the loop doesn't double-append.
    """
    def patched_parse_model(d, ch, verbose=True):
        import ast
        from copy import deepcopy
        from ultralytics.utils import LOGGER, colorstr
        import ultralytics.nn.modules as _m

        # ── Pre-scan: resolve FastViTBackbone channels ────────────────────
        _fastvit_ch: dict[int, list[int]] = {}
        all_layer_defs = d.get("backbone", []) + d.get("head", [])
        for idx, (f, n, m_str, args) in enumerate(all_layer_defs):
            if m_str == "FastViTBackbone":
                model_name  = args[0] if args else "fastvit_sa12"
                out_indices = list(args[2]) if len(args) > 2 else [1, 2, 3]
                if model_name in FASTVIT_CHANNELS:
                    all_c = FASTVIT_CHANNELS[model_name]
                    _fastvit_ch[idx] = [all_c[j] for j in out_indices]
                else:
                    _tmp = FastViTBackbone(model_name, pretrained=False,
                                          out_indices=out_indices)
                    _fastvit_ch[idx] = _tmp.out_channels
                    del _tmp

        if not _fastvit_ch:
            # No FastViTBackbone present — delegate to original
            return original_parse_model(d, ch, verbose)

        # ── We need to run a modified version of parse_model ─────────────
        # Import everything parse_model normally uses
        from ultralytics.nn.modules import (
            AIFI, C1, C2, C2PSA, C2f, C2fAttn, C2fCIB, C2fPSA,
            C3, C3TR, C3Ghost, C3k2, C3x, CBFuse, CBLinear, Classify,
            Concat, Conv, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
            Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, OBB,
            Pose, RepC3, RepNCSPELAN4, ResNetLayer, SCDown, SPPF,
            Segment, WorldDetect, A2C2f,
        )
        # Optional modules (may not exist in all versions)
        for _name in ("AConv", "ADown", "ELAN1", "SPPELAN", "Index"):
            try:
                exec(f"from ultralytics.nn.modules import {_name}", globals())
            except ImportError:
                pass

        Index = globals().get("Index", None)

        nc, act, scales = (d.get(x) for x in ("nc", "activation", "scales"))
        depth, width, kpt_shape = (
            d.get(x, 1.0) for x in ("depth_multiple", "width_multiple", "kpt_shape")
        )
        max_channels = float("inf")
        if scales:
            scale = d.get("scale")
            if not scale:
                scale = tuple(scales.keys())[0]
                LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
            depth, width, max_channels = scales[scale]

        if act:
            Conv.default_act = eval(act)
            if verbose:
                LOGGER.info(f"{colorstr('activation:')} {act}")
        if verbose:
            LOGGER.info(
                f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  "
                f"{'module':<45}{'arguments':<30}"
            )

        ch = [ch]
        layers, save, c2 = [], [], ch[-1]

        base_modules = frozenset({
            Classify, Conv, ConvTranspose, GhostConv, C2fPSA, C2PSA,
            DWConv, Focus, C1, C2, C2f, C3k2, C3, C3TR, C3Ghost,
            DWConvTranspose2d, C3x, RepC3, SCDown, C2fCIB, A2C2f,
        })
        repeat_modules = frozenset({
            C1, C2, C2f, C3k2, C2fAttn, C3, C3TR, C3Ghost,
            C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f,
        })

        for i, (f, n, m_str, args) in enumerate(all_layer_defs):
            # Resolve module
            m = eval(m_str, {
                **globals(),
                **vars(nn),
                "FastViTBackbone": FastViTBackbone,
                **({"Index": Index} if Index else {}),
            })

            n_ = max(round(n * depth), 1) if n > 1 else n

            # ── FastViTBackbone ──────────────────────────────────────────
            if m is FastViTBackbone:
                stage_chs = _fastvit_ch[i]
                m_ = m(*args)
                m_.i, m_.f, m_.type = i, f, m_str
                # Register ALL stage channels so subsequent layers can look them up
                for sc in stage_chs:
                    ch.append(sc)
                c2 = stage_chs[-1]
                np_ = sum(p.numel() for p in m_.parameters())
                if f != -1:
                    save.extend(
                        x % i for x in ([f] if isinstance(f, int) else f) if x != -1
                    )
                layers.append(m_)
                if verbose:
                    LOGGER.info(
                        f"{i:>3}{str(f):>20}{n_:>3}{np_:>10.0f}  "
                        f"{m_str:<45}{str(args):<30}"
                    )
                continue  # ch already appended above — skip bottom append

            # ── Index (picks one stage from backbone list) ───────────────
            if Index is not None and m is Index:
                stage_idx = args[0]
                # Backbone appended len(stages) entries to ch.
                # If backbone is at layer f, it occupies ch[f+1 .. f+n_stages].
                # But we only know ch positions relative to current length.
                # Simpler: the backbone at layer `f` registered its channels
                # into ch starting at position (f+1) in ch[].
                bb_stages = _fastvit_ch.get(f, [])
                if bb_stages:
                    c2 = bb_stages[stage_idx]
                else:
                    c2 = ch[f + 1 + stage_idx]  # fallback
                m_ = m(*args)
                m_.i, m_.f, m_.type = i, f, m_str
                np_ = sum(p.numel() for p in m_.parameters())
                save.extend(
                    x % i for x in ([f] if isinstance(f, int) else f) if x != -1
                )
                layers.append(m_)
                ch.append(c2)
                if verbose:
                    LOGGER.info(
                        f"{i:>3}{str(f):>20}{n_:>3}{np_:>10.0f}  "
                        f"{m_str:<45}{str(args):<30}"
                    )
                continue

            # ── Standard Ultralytics layer handling ──────────────────────
            args_ = list(args)
            if isinstance(f, int):
                c1 = ch[f]
            elif isinstance(f, list):
                c1 = ch[f[-1]]
            else:
                c1 = ch[-1]

            if m in base_modules:
                c2 = args[0]
                if c2 != nc:
                    c2 = min(int(c2 * width), int(max_channels))
                if m in repeat_modules:
                    args_ = [c1, c2, *args[1:]]
                    if m in {C2f, C3k2, C2fAttn, C3, C3Ghost, C3TR,
                             C3x, RepC3, C2fPSA, C2fCIB, C2PSA, A2C2f}:
                        args_.insert(2, n_)
                        n_ = 1
                else:
                    args_ = [c1, c2, *args[1:]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
                args_ = list(args)
            elif m in {Detect, Segment, Pose, OBB, WorldDetect}:
                args_ = list(args)
                args_.insert(1, [ch[x] for x in f])
                c2 = args[0]
            elif m is nn.BatchNorm2d:
                args_ = [c1]
                c2 = c1
            else:
                c2 = ch[f] if isinstance(f, int) else c1

            m_ = nn.Sequential(*(m(*args_) for _ in range(n_))) if n_ > 1 else m(*args_)
            m_.i, m_.f, m_.type = i, f, m_str
            np_ = sum(p.numel() for p in m_.parameters())
            save.extend(
                x % i for x in ([f] if isinstance(f, int) else f) if x != -1
            )
            layers.append(m_)
            ch.append(c2)
            if verbose:
                LOGGER.info(
                    f"{i:>3}{str(f):>20}{n_:>3}{np_:>10.0f}  "
                    f"{m_str:<45}{str(args_):<30}"
                )

        return nn.Sequential(*layers), sorted(save)

    return patched_parse_model


def patch_ultralytics() -> None:
    """
    Inject FastViTBackbone into Ultralytics and patch parse_model.

    MUST be called before `from ultralytics import YOLO` or any YOLO() call.

    Example
    -------
        from fastvit_backbone import patch_ultralytics
        patch_ultralytics()

        from ultralytics import YOLO
        model = YOLO("yolo26-fastvit.yaml")
        model.train(data="coco.yaml", epochs=300)
    """
    import ultralytics.nn.tasks   as _tasks
    import ultralytics.nn.modules as _modules

    # 1. Make FastViTBackbone visible in both namespaces
    _tasks.FastViTBackbone   = FastViTBackbone
    _modules.FastViTBackbone = FastViTBackbone

    # 2. Replace parse_model with our channel-aware version
    _tasks.parse_model = _build_patched_parse_model(_tasks.parse_model)

    print(
        "[patch_ultralytics] ✓ FastViTBackbone registered.\n"
        "[patch_ultralytics] ✓ parse_model patched for multi-output backbone channels."
    )