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


class MultiHeadGate(nn.Module):
    """
    Multi-headed soft spatial gate.
    
    Instead of a single scalar importance score per token, uses multiple
    gate heads with softmax normalization across heads. This allows the
    gate to capture multiple independent notions of token importance
    (e.g., foreground vs background, different object scales).
    
    Args:
        dim (int): Channel dimension
        num_heads (int): Number of independent gate heads
        init_alpha (float): Initial gate blend factor (sigmoid(-5) ≈ 0.007)
    """
    def __init__(self, dim, num_heads=4, init_alpha=-5.0):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.proj = nn.Linear(dim, num_heads, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.gate_alpha = nn.Parameter(torch.full((1,), init_alpha))

    def forward(self, x):
        B, C, H, W = x.shape
        flat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        scores = self.proj(self.norm(flat))  # [B, N, H]
        scores = torch.clamp(scores, -6.0, 6.0)
        gate = torch.sigmoid(scores)  # [B, N, H], each head ∈ (0,1)
        gate = gate.mean(dim=-1, keepdim=True)  # [B, N, 1] — averaged over heads
        alpha = torch.sigmoid(self.gate_alpha)
        gated = flat * (1.0 + 2.0 * (gate - 0.5))
        out = flat + alpha * (gated - flat)
        return out.reshape(B, H, W, C).permute(0, 3, 1, 2)


class DynamicPSABlock(nn.Module):
    """
    Enhanced PSABlock with multi-headed soft spatial gating.
    
    Architecture:
      x → pre_gate → Attention → post_gate → FFN → out
      
    The pre-attention gate filters irrelevant tokens before attention,
    while the post-FFN gate (original design) refines the output.
    Both gates use multi-headed scoring for richer importance estimation.
    
    Args:
        c (int): Channel dimension
        attn_ratio (float): Key/query dimension ratio for Attention
        num_heads (int): Number of attention heads
        num_gate_heads (int): Number of gate heads
        shortcut (bool): Use residual connections
    """
    def __init__(self, c, attn_ratio=0.5, num_heads=4, num_gate_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

        self.pre_gate = MultiHeadGate(c, num_heads=num_gate_heads, init_alpha=-5.0)
        self.post_gate = MultiHeadGate(c, num_heads=num_gate_heads, init_alpha=-5.0)

    def forward(self, x):
        gated = self.pre_gate(x)
        x = x + self.attn(gated) if self.add else self.attn(gated)
        x = self.post_gate(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class DynamicTransformerBlock(nn.Module):
    """
    Enhanced DynamicTransformerBlock replacing C2PSA in YOLO26-RL.
    
    Follows the proven C2PSA pattern:
      1. cv1 splits channels into (skip, process) streams
      2. Process stream goes through N × DynamicPSABlock
         (multi-headed pre-gate → Attention → multi-headed post-gate → FFN)
      3. cv2 merges streams back
    
    Unlike the original, this version:
      - Respects the YAML repeat count for number of blocks
      - Uses multi-headed gating (richer token importance)
      - Has pre-attention gating (filters before attention)
      - Supports higher head counts for better capacity
    
    YAML format: [c2, num_heads, num_gate_heads]
    """
    def __init__(self, c1, c2, n=1, num_heads=4, num_gate_heads=4, *args):
        super().__init__()
        assert c1 == c2, f"DynamicTransformerBlock requires c1 == c2, got {c1} != {c2}"

        self.c1 = c1
        self.c2 = c2
        self.hidden_c = c2 // 2

        self.cv1 = Conv(c1, 2 * self.hidden_c, 1, 1)
        self.m = nn.Sequential(*[
            DynamicPSABlock(
                self.hidden_c, attn_ratio=0.5,
                num_heads=num_heads,
                num_gate_heads=num_gate_heads,
                shortcut=True,
            )
            for _ in range(n)
        ])
        self.cv2 = Conv(2 * self.hidden_c, c2, 1)

    def forward(self, x):
        a, b = self.cv1(x).split((self.hidden_c, self.hidden_c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))
