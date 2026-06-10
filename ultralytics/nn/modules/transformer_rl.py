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
    with a numerically stable soft spatial gate.

    Uses the exact same Attention + FFN pattern as PSABlock for compatibility.
    The soft gate uses residual formulation: output = x + alpha * (gate * x - x)
    where alpha is initialized near 0 so the gate starts as identity.
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

        # Soft spatial gate — residual formulation (NaN-safe)
        # gate_weight initialized near 0 → block starts as pure PSABlock
        self.gate_norm = nn.LayerNorm(c, eps=1e-6)
        self.gate_proj = nn.Linear(c, 1, bias=True)
        # Initialize gate_proj to output near-zero → sigmoid(~0) ≈ 0.5 → gate_delta ≈ 0
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)
        # Learnable blend factor: sigmoid(-5)≈0.007 → gate is ~identity at init
        # The block starts as a pure PSABlock and gradually learns gating
        self.gate_alpha = nn.Parameter(torch.full((1,), -5.0))

    def forward(self, x):
        B, C, H, W = x.shape

        # --- Same as PSABlock ---
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)

        # --- Numerically stable soft spatial gate ---
        # Residual: out = x + alpha * (gated_x - x)
        # When alpha=0 → out=x (identity, safe start)
        # When alpha→1 → out = gated_x (full gate effect)
        alpha = torch.sigmoid(self.gate_alpha)  # ∈ (0, 1)

        flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, N, C]
        scores = self.gate_proj(self.gate_norm(flat)).squeeze(-1)  # [B, N]
        # Clamp scores to prevent extreme sigmoid inputs
        scores = torch.clamp(scores, -6.0, 6.0)
        gate = torch.sigmoid(scores).unsqueeze(-1)  # [B, N, 1] ∈ (0, 1)

        # Gated version: scale each token by its gate value
        gated_flat = flat * (1.0 + 2.0 * (gate - 0.5))  # gate=0.5 → x1.0, gate=1→x2, gate=0→x0
        # Residual blend: starts at x (alpha=0), gradually learns gating
        out_flat = flat + alpha * (gated_flat - flat)

        x = out_flat.reshape(B, H, W, C).permute(0, 3, 1, 2)

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
