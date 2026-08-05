# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.nn.modules.head import Detect
from ultralytics.utils.torch_utils import copy_attr

from .tasks import load_checkpoint


class FeatureHook:
    """Picklable forward hook that stores layer output into a shared dict."""

    def __init__(self, feat_dict: dict, idx: int) -> None:
        """Initialize the hook with the shared feature dict and the layer index to store outputs under."""
        self.feat_dict = feat_dict
        self.idx = idx

    def __call__(self, module: nn.Module, inputs: tuple, output) -> None:
        """Store the layer's forward output into the shared feature dict under its index.

        The output is a tensor for neck layers but a tuple/dict for the Detect head, so it is left untyped.
        """
        self.feat_dict[self.idx] = output


class DistillationModel(nn.Module):
    """YOLO knowledge distillation model.

    This class wraps a teacher-student pair for knowledge distillation training. Features are extracted from both models
    via forward hooks for distillation.

    Attributes:
        teacher_model (nn.Module): Frozen teacher model providing features.
        student_model (nn.Module): Trainable student model being distilled.
        feats_idx (list): Layer indices for feature extraction.
        projector (nn.ModuleList): MLP projector aligning student features to teacher dimensions.
        dis (float): Distillation loss weight factor.

    Methods:
        get_distill_layers: Auto-detect distillation feature layers from the Detect head.
        forward: Run the student model, or compute the combined loss when given a training batch.
        loss: Compute combined detection and distillation loss.
        loss_sl2: Compute score-weighted L2 distillation loss for a feature pair.
        decouple_outputs: Normalize teacher/student head outputs across train/val formats.
        train: Set training mode while keeping teacher frozen.

    Examples:
        Train a student model with knowledge distillation from a larger teacher (the trainer builds the
        DistillationModel internally when the ``distill_model`` argument is set)
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> model.train(data="coco8.yaml", distill_model="yolo26s.pt")
    """

    def __init__(self, teacher_model: str | Path | nn.Module, student_model: nn.Module):
        """Initialize the distillation model with teacher, student, and feature extraction hooks.

        Args:
            teacher_model (str | Path | nn.Module): Teacher model checkpoint path or module.
            student_model (nn.Module): Student model module to be trained.
        """
        super().__init__()
        ch = student_model.yaml.get("channels", 3)
        if isinstance(teacher_model, (str, Path)):
            teacher_model = load_checkpoint(teacher_model)[0]
            if teacher_model.yaml.get("channels", 3) != ch:
                weights = teacher_model
                teacher_model = type(weights)(weights.yaml.copy(), ch=ch, nc=weights.yaml["nc"], verbose=False)
                teacher_model.load(weights)
        device = next(student_model.parameters()).device
        self.teacher_model = teacher_model.to(device)
        self._freeze_teacher()
        self.student_model = student_model
        self.feats_idx = self.get_distill_layers(student_model)

        # Hook-based feature capture: identical for teacher and student
        self._teacher_feats: dict[int, torch.Tensor] = {}
        self._student_feats: dict[int, torch.Tensor] = {}
        self._teacher_hooks: list = []
        self._student_hooks: list = []
        self._register_feature_hooks()

        # Get feature dimensions via dummy forward pass (hooks capture outputs)
        imgsz = student_model.args.imgsz
        student_model.eval()
        with torch.no_grad():
            im = torch.zeros(2, ch, imgsz, imgsz, device=device)
            teacher_model(im)
            student_model(im)
        student_model.train()
        teacher_output = [self._teacher_feats[idx] for idx in self.feats_idx]
        student_output = [self._student_feats[idx] for idx in self.feats_idx]

        copy_attr(self, student_model)
        self.dis = self.student_model.args.dis
        projectors = []
        for student_out, teacher_out in zip(student_output[:-1], teacher_output[:-1]):
            student_dim = self.decouple_outputs(student_out).shape[1]
            teacher_dim = self.decouple_outputs(teacher_out).shape[1]
            projectors.append(
                nn.Sequential(
                    nn.Conv2d(student_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(teacher_dim, teacher_dim, kernel_size=1, stride=1, padding=0),
                )
            )
        self.projector = nn.ModuleList(projectors).to(device)

    def __getstate__(self):
        """Return a copy of state for pickling without captured features or hook handles.

        Clears the feature dicts in place (rather than replacing the attributes) because the registered
        FeatureHooks share these exact dict objects; otherwise a deepcopy/pickle of a mid-training model would
        still reach the hook-held tensors (which carry grad_fn and cannot be deep-copied).
        """
        self._teacher_feats.clear()
        self._student_feats.clear()
        state = self.__dict__.copy()
        state["_teacher_hooks"] = []
        state["_student_hooks"] = []
        return state

    def __setstate__(self, state):
        """Clear stale features and hooks, and re-register forward hooks after unpickling."""
        self.__dict__.update(state)
        self._teacher_feats = {}
        self._student_feats = {}
        self._register_feature_hooks()

    def _remove_feature_hooks(self) -> None:
        """Remove any previously registered feature-capture hooks."""
        for handle in self._student_hooks:
            handle.remove()
        self._student_hooks.clear()
        if self.teacher_model is not None:
            for handle in self._teacher_hooks:
                handle.remove()
            self._teacher_hooks.clear()

    @staticmethod
    def _clear_feature_hooks(module: nn.Module) -> None:
        """Remove any FeatureHook instances from a module's forward hooks."""
        for handle_id, hook in list(module._forward_hooks.items()):
            if isinstance(hook, FeatureHook):
                del module._forward_hooks[handle_id]

    def _register_feature_hooks(self) -> None:
        """Register feature-capture hooks, removing stale FeatureHook instances first."""
        self._remove_feature_hooks()
        for idx in self.feats_idx:
            self._clear_feature_hooks(self.student_model.model[idx])
            self._student_hooks.append(
                self.student_model.model[idx].register_forward_hook(FeatureHook(self._student_feats, idx))
            )
            if self.teacher_model is not None:
                self._clear_feature_hooks(self.teacher_model.model[idx])
                self._teacher_hooks.append(
                    self.teacher_model.model[idx].register_forward_hook(FeatureHook(self._teacher_feats, idx))
                )

    @staticmethod
    def get_distill_layers(model: nn.Module) -> list[int]:
        """Auto-detect distillation feature layers from the model's Detect head.

        Returns the Detect head's input layer indices plus the head layer index itself.
        E.g. YOLO26 -> [16, 19, 22, 23], YOLOv8 -> [15, 18, 21, 22].
        """
        for m in model.model:
            if isinstance(m, Detect):
                return [*list(m.f), m.i]
        raise ValueError("No Detect head found in model")

    def _freeze_teacher(self):
        """Keep teacher fixed for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set model train mode while keeping teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """Compute loss.

        Args:
            batch (dict): Batch to compute loss on.
            preds (torch.Tensor | list[torch.Tensor], optional): Predictions.
        """
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        # Clear feature dicts before forward passes
        self._teacher_feats.clear()
        self._student_feats.clear()

        with torch.no_grad():
            self.teacher_model(batch["img"])  # hooks capture teacher features
        preds = self.student_model(batch["img"])  # hooks capture student features

        regular_loss, loss_items = self.student_model.loss(batch, preds)
        teacher_head_feat = self._teacher_feats[self.feats_idx[-1]]
        teacher_scores = (
            self.decouple_outputs(teacher_head_feat, branch="one2many")["scores"]
            + self.decouple_outputs(teacher_head_feat, branch="one2one")["scores"]
        ) / 2
        # neck feature sizes vary per batch (e.g. multi_scale), so split scores by the live teacher feats
        neck_feats = [self._teacher_feats[idx] for idx in self.feats_idx[:-1]]
        parts = torch.split(teacher_scores, [f.shape[-2] * f.shape[-1] for f in neck_feats], dim=-1)
        teacher_scores = tuple(p.sigmoid().max(dim=1, keepdim=True).values for p in parts)
        for i, feat_idx in enumerate(self.feats_idx[:-1]):
            teacher_feat = self.decouple_outputs(self._teacher_feats[feat_idx])
            student_feat = self.projector[i](self.decouple_outputs(self._student_feats[feat_idx]))
            loss_distill += (
                self.loss_sl2(student_feat, teacher_feat, feat_idx=i, teacher_scores=teacher_scores) * self.dis
            )

        loss_items["dis_loss"] = loss_distill.detach()
        loss_distill = loss_distill * batch["img"].shape[0]
        return torch.cat([regular_loss, loss_distill]), loss_items

    def loss_sl2(
        self, student_feat: torch.Tensor, teacher_feat: torch.Tensor, feat_idx: int, teacher_scores: tuple
    ) -> torch.Tensor:
        """Compute score-weighted L2 distillation loss for a feature pair.

        Args:
            student_feat (torch.Tensor): Student feature tensor of shape (N, C, H, W).
            teacher_feat (torch.Tensor): Teacher feature tensor of shape (N, C, H, W).
            feat_idx (int): Index of the feature level for selecting teacher scores.
            teacher_scores (tuple): Tuple of score tensors for each feature level.

        Returns:
            (torch.Tensor): The computed score-weighted L2 loss.
        """
        teacher_score = teacher_scores[feat_idx]
        n, c = student_feat.shape[:2]
        student_feat = student_feat.view(n, c, -1)
        teacher_feat = teacher_feat.view(n, c, -1)
        mse = F.mse_loss(student_feat, teacher_feat, reduction="none")
        weighted_mse = (mse * teacher_score).sum() / (teacher_score.sum() * c + 1e-9)
        return weighted_mse

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)

    def decouple_outputs(self, preds, branch: str = "one2one"):
        """Decouple outputs for teacher/student models.

        This method handles different output formats from YOLO models, including
        tuple outputs (train/val mode), dict outputs with branches (one2one/one2many),
        and direct tensor outputs.

        Args:
            preds (torch.Tensor | tuple | dict): Model predictions in various formats.
            branch (str): Which branch to extract from dict outputs ("one2one" or "one2many").

        Returns:
            (torch.Tensor | dict): The decoupled predictions.
        """
        if isinstance(preds, tuple):  # decouple for val mode
            preds = preds[1]
        if isinstance(preds, dict):
            if branch in preds:
                preds = preds[branch]
        return preds


class DINOv3Teacher(nn.Module):
    """DINOv3 vision-foundation-model teacher (RT-DETRv4 recipe).

    Loads a frozen DINOv3 ViT-B/16 model via ``torch.hub`` from a local clone of
    the facebookresearch/dinov3 repo plus a downloaded checkpoint.  The input is
    2x2 avg-pooled then patchified at 16px, so the returned patch-token map sits
    at 1/32 of the image resolution — matching a student's stride-32 feature.
    """

    def __init__(self, repo_path: str, weights_path: str, model_type: str = "dinov3_vitb16"):
        """Initialize the frozen DINOv3 teacher.

        Args:
            repo_path (str): Path to a local clone of the dinov3 repo (torch.hub source='local').
            weights_path (str): Path to the DINOv3 checkpoint (.pth).
            model_type (str): torch.hub entry name, e.g. ``dinov3_vitb16``.
        """
        super().__init__()
        self.model = torch.hub.load(repo_path, model_type, source="local", weights=weights_path)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.teacher_feature_dim = getattr(self.model, "embed_dim", 768)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.avgpool = nn.AvgPool2d(2, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return the teacher patch-token feature map (B, C, H, W) at 1/32 resolution.

        Args:
            images (torch.Tensor): Normalized-ish input batch (values in [0, 1]).

        Returns:
            (torch.Tensor): Teacher feature map.
        """
        x = self.avgpool((images - self.mean) / self.std)
        with torch.no_grad():
            out = self.model(x, is_training=True, masks=None)
        tokens = out["x_norm_patchtokens"]
        b, n, c = tokens.shape
        h = w = int(n**0.5)
        return tokens.permute(0, 2, 1).reshape(b, c, h, w)


class DINOv3DistillationModel(nn.Module):
    """RT-DETRv4-style DINOv3 feature distillation wrapper for LightEdgeDet.

    A frozen DINOv3 ViT-B/16 teacher supervises the student's stride-32 backbone
    feature through a 1x1 project head using the cosine distance (``1 - cos_sim``)
    weighted by ``args.dis`` — the same loss RT-DETRv4 applies.  The teacher is
    only present at training time and is stripped on inference.
    """

    TEACHER_DIM = 768

    def __init__(self, teacher_model, student_model):
        """Wrap the student with the DINOv3 teacher and a 1x1 project head.

        Args:
            teacher_model: DINOv3 teacher config — a dict or JSON string with keys
                ``repo``, ``weights`` and optional ``model_type``, or a ready nn.Module.
            student_model (nn.Module): LightEdgeDetModel to distill.
        """
        super().__init__()
        self.student_model = student_model
        self.distill_feat_idx = int(getattr(student_model, "distill_feat_idx", 2))
        self.teacher_model = self._build_teacher(teacher_model)
        device = next(student_model.parameters()).device
        self.teacher_model = self.teacher_model.to(device)
        self._freeze_teacher()

        # Student feature width for the project head: read from a real forward so the
        # index stays correct regardless of the backbone's out_indices configuration.
        student_model.eval()
        with torch.no_grad():
            dummy = student_model.backbone(
                torch.zeros(1, student_model.backbone.stem[0].in_channels, 128, 128, device=device)
            )
        student_c = dummy[self.distill_feat_idx].shape[1]
        student_model.train()

        self.projector = nn.Sequential(
            nn.Conv2d(student_c, self.TEACHER_DIM, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.TEACHER_DIM, self.TEACHER_DIM, 1),
        ).to(device)

        copy_attr(self, student_model)
        self.dis = float(getattr(student_model.args, "dis", 6.0))

    @staticmethod
    def _build_teacher(teacher_model):
        """Build a frozen DINOv3Teacher from a dict/JSON config or reuse a module."""
        if isinstance(teacher_model, nn.Module):
            return teacher_model
        cfg = teacher_model if isinstance(teacher_model, dict) else json.loads(str(teacher_model))
        return DINOv3Teacher(
            repo_path=cfg["repo"],
            weights_path=cfg["weights"],
            model_type=cfg.get("model_type", "dinov3_vitb16"),
        )

    def _freeze_teacher(self):
        """Keep the teacher fixed for distillation."""
        if self.teacher_model is None:
            return
        self.teacher_model.eval()
        for v in self.teacher_model.parameters():
            if v.requires_grad:
                v.requires_grad = False

    def train(self, mode: bool = True):
        """Set the model to train mode while keeping the teacher frozen in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def forward(self, x, *args, **kwargs):
        """Forward pass through the student model."""
        if isinstance(x, dict):  # for cases of training and validating while training.
            return self.loss(x, *args, **kwargs)
        return self.student_model.predict(x, *args, **kwargs)

    def loss(self, batch, preds=None):
        """Compute the combined detection + DINOv3 distillation loss."""
        loss_distill = torch.zeros(1, device=batch["img"].device)
        if not self.training:  # for loss calculation during validation while training
            if preds is None:
                preds = self.student_model(batch["img"])
            regular_loss, loss_items = self.student_model.loss(batch, preds)
            loss_items["dis_loss"] = loss_distill.detach()
            return torch.cat([regular_loss, loss_distill]), loss_items

        img = batch["img"]
        feats = self.student_model.backbone(img)  # [P3, P4, P5, P6]
        student_feat = self.projector(feats[self.distill_feat_idx])

        with torch.no_grad():
            teacher_feat = self.teacher_model(img)
        if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
            teacher_feat = F.interpolate(
                teacher_feat, size=student_feat.shape[-2:], mode="bilinear", align_corners=False
            )

        s = F.normalize(student_feat.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        t = F.normalize(teacher_feat.flatten(2).permute(0, 2, 1), p=2, dim=-1)
        loss_distill = ((1 - F.cosine_similarity(s, t, dim=-1)).mean() * self.dis).reshape(1)

        preds = self.student_model.detect(self.student_model.neck(feats))
        regular_loss, loss_items = self.student_model.loss(batch, preds)
        loss_items["dis_loss"] = loss_distill.detach()
        return torch.cat([regular_loss, loss_distill * img.shape[0]]), loss_items

    @property
    def criterion(self):
        """Get the criterion from the student model."""
        return self.student_model.criterion

    @criterion.setter
    def criterion(self, value) -> None:
        """Set value for student criterion."""
        self.student_model.criterion = value

    def init_criterion(self):
        """Initialize the loss criterion via the student model."""
        return self.student_model.init_criterion()

    @property
    def end2end(self):
        """Expose student end-to-end mode for validator/predictor control."""
        return getattr(self.student_model, "end2end", False)

    @end2end.setter
    def end2end(self, value):
        """Forward end-to-end mode update to the student model."""
        self.student_model.end2end = value

    def set_head_attr(self, **kwargs):
        """Forward head-attribute updates (e.g. max_det, agnostic_nms, end2end) to the student model."""
        self.student_model.set_head_attr(**kwargs)
