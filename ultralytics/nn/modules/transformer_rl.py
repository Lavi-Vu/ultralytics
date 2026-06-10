import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv
from ultralytics.nn.modules.block import Attention


class RLPruningController(nn.Module):
    """
    RL policy network that predicts a pruning ratio from a global image descriptor.
    Kept for backward compatibility; not used by the improved DynamicTransformerBlock.
    """
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class DynamicPSABlock(nn.Module):
    """
    Enhanced PSABlock that mirrors the proven YOLO26 PSABlock architecture
    but adds a soft spatial gate for dynamic token weighting.

    Uses the exact same Attention + FFN pattern as PSABlock for compatibility.
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

        # Soft spatial gate (lightweight, differentiable)
        # Scores each spatial token and re-weights — no tokens are destroyed
        self.gate_norm = nn.LayerNorm(c)
        self.gate_proj = nn.Linear(c, 1)
        self.gate_temp = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Same as PSABlock ---
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)

        # --- Soft spatial gate ---
        flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]
        scores = self.gate_proj(self.gate_norm(flat)).squeeze(-1)  # [B, N]
        temp = torch.clamp(self.gate_temp.abs(), 0.1, 5.0)
        gate = torch.sigmoid(scores / temp).unsqueeze(-1)  # [B, N, 1]
        # Normalize so mean gate ≈ 1 (preserves activation magnitude)
        gate = gate / (gate.mean(dim=1, keepdim=True).detach() + 1e-6)
        flat = flat * gate
        x = flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


class DynamicTransformerBlock(nn.Module):
    """
    Improved DynamicTransformerBlock that replaces C2PSA in YOLO26.

    Architecture follows the proven C2PSA pattern exactly:
      1. cv1 splits channels into (skip, process) streams
      2. Process stream goes through N x DynamicPSABlock
         (Attention + FFN + soft spatial gate)
      3. cv2 merges streams back

    The soft spatial gate in each DynamicPSABlock provides a
    differentiable, information-preserving alternative to hard
    token pruning. It learns to weight spatial positions by
    importance without destroying any information.

    This replaces the original block which had:
      - Hard token pruning (destroyed spatial info → big mAP drop)
      - No FFN (missing key feature transform)
      - No channel split (poor gradient flow)
      - Mean-expand restoration (all tokens identical → collapse)
    """
    def __init__(self, c1, c2, *args):
        super().__init__()
        # c2: output channels (first arg in YAML list)
        # args[0]: num_heads (default follows C2PSA convention)
        # Remaining args are ignored for YAML compatibility

        assert c1 == c2, f"DynamicTransformerBlock requires c1 == c2, got {c1} != {c2}"

        self.c1 = c1
        self.c2 = c2
        self.num_heads = args[0] if len(args) > 0 else 4

        # Hidden channel count — same ratio as C2PSA (e=0.5)
        self.hidden_c = c2 // 2

        # ---- Channel split via 1x1 conv (identical to C2PSA) ----
        self.cv1 = Conv(c1, 2 * self.hidden_c, 1, 1)

        # ---- Stack of DynamicPSABlocks ----
        # Use 2 blocks to match C2PSA n=2 in baseline
        self.m = nn.Sequential(
            DynamicPSABlock(self.hidden_c, attn_ratio=0.5,
                            num_heads=self.num_heads, shortcut=True),
            DynamicPSABlock(self.hidden_c, attn_ratio=0.5,
                            num_heads=self.num_heads, shortcut=True),
        )

        # ---- Output merge (identical to C2PSA) ----
        self.cv2 = Conv(2 * self.hidden_c, c2, 1)

    def forward(self, x):
        """x: [B, C, H, W] → [B, C, H, W]"""
        # Split into skip and process branches (same as C2PSA)
        a, b = self.cv1(x).split((self.hidden_c, self.hidden_c), dim=1)
        # Process only branch b through transformer blocks
        b = self.m(b)
        # Merge and project
        return self.cv2(torch.cat((a, b), dim=1))
