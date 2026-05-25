# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Linear Attention modules for efficient YOLO implementations."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv


class LinearAttention(nn.Module):
    """
    Linear Attention module that approximates softmax attention with linear complexity.

    Based on the Linear Transformer architecture, this replaces the
    standard quadratic softmax attention with kernel-based feature maps that enable
    linear time and space complexity.

    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        attn_ratio (float): Attention ratio for key dimension.
        feature_map (str): Type of feature map ('relu', 'elu').
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5,
                 feature_map: str = 'elu', eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = max(1, int(self.head_dim * attn_ratio))
        self.value_dim = self.key_dim  # use attn_ratio for value too (saves params)
        self.feature_map = feature_map
        self.eps = eps

        # Projections for query, key, value
        self.q_proj = Conv(dim, self.key_dim * num_heads, 1, act=False)
        self.k_proj = Conv(dim, self.key_dim * num_heads, 1, act=False)
        self.v_proj = Conv(dim, self.value_dim * num_heads, 1, act=False)
        self.out_proj = Conv(self.value_dim * num_heads, dim, 1, act=False)

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map to approximate softmax kernel."""
        if self.feature_map == 'relu':
            return F.relu(x)
        elif self.feature_map == 'elu':
            return F.elu(x) + 1
        else:
            raise ValueError(f"Unsupported feature map: {self.feature_map}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Linear Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        N = H * W

        # Project to query, key, value
        q = self.q_proj(x).view(B, self.num_heads, self.key_dim, N)  # (B, heads, key_dim, N)
        k = self.k_proj(x).view(B, self.num_heads, self.key_dim, N)  # (B, heads, key_dim, N)
        v = self.v_proj(x).view(B, self.num_heads, self.value_dim, N)  # (B, heads, value_dim, N)

        # Apply feature maps
        q = self._feature_map(q)  # (B, heads, key_dim, N)
        k = self._feature_map(k)  # (B, heads, key_dim, N)
        # v stays as is

        # Compute linear attention with normalization
        # KV^T: (B, heads, key_dim, value_dim)
        kv = torch.einsum('bhkn,bhvm->bhkvm', k, v)

        # Q(KV): (B, heads, key_dim, N) @ (B, heads, key_dim, value_dim) -> (B, heads, value_dim, N)
        out = torch.einsum('bhkn,bhkvm->bhvm', q, kv)

        # Normalization: 1 / (Q * K^T * 1) where we sum over key_dim
        # For each position, we want sum over key_dim of (q * k)
        # q: (B, heads, key_dim, N), k: (B, heads, key_dim, N)
        # Result: (B, heads, N)
        z = 1 / (torch.einsum('bhkn,bhkn->bhn', q, k) + self.eps)  # (B, heads, N)

        out = out * z.unsqueeze(2)  # (B, heads, value_dim, N)

        # Reshape and project back
        out = out.contiguous().view(B, self.value_dim * self.num_heads, H, W)
        out = self.out_proj(out)

        return out


class PerformerAttention(nn.Module):
    """
    Performer-style attention using orthogonal random features for kernel approximation.

    This implements the Performer approach which uses FAVOR+ (Fast Attention Via
    Orthogonal Random features) to approximate the softmax kernel with linear complexity.

    Args:
        dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        attn_ratio (float): Attention ratio for key dimension.
        nb_features (int): Number of random features for approximation.
        generalized_attention (bool): Whether to use generalized attention.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5,
                 nb_features: int = 32, generalized_attention: bool = True, eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.value_dim = self.head_dim
        self.nb_features = nb_features
        self.generalized_attention = generalized_attention
        self.eps = eps

        # Projections
        self.q_proj = Conv(dim, self.key_dim * num_heads, 1, act=False)
        self.k_proj = Conv(dim, self.key_dim * num_heads, 1, act=False)
        self.v_proj = Conv(dim, self.value_dim * num_heads, 1, act=False)
        self.out_proj = Conv(self.value_dim * num_heads, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

        # Random features for kernel approximation
        self.register_parameter('w_q', nn.Parameter(torch.randn(num_heads, nb_features, self.key_dim) / math.sqrt(self.key_dim)))
        self.register_parameter('w_k', nn.Parameter(torch.randn(num_heads, nb_features, self.key_dim) / math.sqrt(self.key_dim)))

    def _feature_map(self, x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Apply random feature map for kernel approximation."""
        if self.generalized_attention:
            return F.elu(x) + 1
        else:
            return F.relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Performer Attention.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        B, C, H, W = x.shape
        N = H * W

        # Project to query, key, value
        q = self.q_proj(x).view(B, self.num_heads, self.key_dim, N)  # (B, heads, key_dim, N)
        k = self.k_proj(x).view(B, self.num_heads, self.key_dim, N)  # (B, heads, key_dim, N)
        v = self.v_proj(x).view(B, self.num_heads, self.value_dim, N)  # (B, heads, value_dim, N)

        # Apply random feature maps
        # q shape: (B, heads, key_dim, N)
        # w_q shape: (heads, nb_features, key_dim)
        # q_prime shape: (B, heads, nb_features, N)
        q_prime = torch.einsum('b h k n, h m k -> b h m n', q, self.w_q)  # (B, heads, nb_features, N)
        k_prime = torch.einsum('b h k n, h m k -> b h m n', k, self.w_k)  # (B, heads, nb_features, N)

        if self.generalized_attention:
            q_prime = F.elu(q_prime) + 1
            k_prime = F.elu(k_prime) + 1
        else:
            q_prime = F.relu(q_prime)
            k_prime = F.relu(k_prime)

        # Compute Performer attention: Q'(K'^T V) / (Q' * 1^T K')
        # First compute K'^T V: (B, heads, nb_features, value_dim)
        kv = torch.einsum('b h m n, b h v n -> b h m v', k_prime, v)  # (B, heads, nb_features, value_dim)
        # Then compute Q'(K'^T V): (B, heads, nb_features, N) @ (B, heads, nb_features, value_dim) -> (B, heads, value_dim, N)
        out = torch.einsum('b h m n, b h m v -> b h v n', q_prime, kv)  # (B, heads, value_dim, N)

        # Normalization term: Q' * 1^T K'
        # Compute: 1 / (Q' * sum_n K'[:, :, :, n]) where sum_n is over N dimension
        ones_N = torch.ones(N, device=x.device, dtype=x.dtype)  # (N)
        k_sum_N = torch.einsum('b h m n, n -> b h m', k_prime, ones_N)  # (B, heads, nb_features)
        # Actually we want: for each position, sum over nb_features of Q' * K'
        qk_sum = torch.einsum('b h m n, b h m n -> b h n', q_prime, k_prime)  # (B, heads, N)
        # For stability, let's compute it as: 1 / (||Q'|| * ||K'|| + eps) per head per position
        # But let's stick to the original formula: 1 / (Q' * 1^T * K' * 1)
        # Q' * 1_N: (B, heads, nb_features)
        q_sum_N = torch.einsum('b h m n, n -> b h m', q_prime, ones_N)  # (B, heads, nb_features)
        # K' * 1_N: (B, heads, nb_features)
        k_sum_N = torch.einsum('b h m n, n -> b h m', k_prime, ones_N)  # (B, heads, nb_features)
        # Now we want: sum over m of q_sum_N * k_sum_N -> (B, heads)
        qk_sum_N = torch.einsum('b h m, b h m -> b h', q_sum_N, k_sum_N)  # (B, heads)
        z = 1 / (qk_sum_N.unsqueeze(-1) + self.eps)  # (B, heads, 1)

        out = out * z.unsqueeze(-1)  # (B, heads, value_dim, N)

        # Reshape and project back
        out = out.contiguous().view(B, self.value_dim * self.num_heads, H, W)
        out = self.out_proj(out)
        out = out + self.pe(x)

        return out


class LinearAttentionBlock(nn.Module):
    """
    Linear Attention Block combining linear attention with feed-forward network.

    Similar to TransformerBlock but uses linear attention for efficiency.
    """

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5,
                 mlp_ratio: float = 0.25, feature_map: str = 'elu', act=None):
        super().__init__()
        self.attn = LinearAttention(dim, num_heads, attn_ratio, feature_map)
        self.mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, self.mlp_dim, 1),
            Conv(self.mlp_dim, dim, 1, act=False)
        )
        self.act = act() if act else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Linear Attention Block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, C, H, W).
        """
        # Self-attention
        x = x + self.attn(x)
        # Feed-forward network
        x = x + self.mlp(x)
        return x


# Export the main classes
__all__ = [
    "LinearAttention",
    "PerformerAttention",
    "LinearAttentionBlock"
]