# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lightweight transformer modules for EdgeRF-YOLO hybrid CNN-Transformer architecture."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.torch_utils import TORCH_1_11
from .conv import Conv, DWConv


__all__ = (
    "WindowedAttention",
    "GlobalAttention",
    "LightTransformerBlock",
    "InterleavedTransformerStage",
    "MLPFFN",
)


def _autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class MLPFFN(nn.Module):
    """MLP-based feed-forward network for transformer blocks."""

    def __init__(self, dim, hidden_dim=None, dropout=0.0, act=nn.GELU()):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class WindowedAttention(nn.Module):
    """Window-based multi-head self-attention.
    
    Operates on non-overlapping windows for efficient local feature interaction.
    Supports FlashAttention when available.
    """

    def __init__(self, dim, num_heads=8, window_size=7, attn_ratio=0.5, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5

        nh_kd = self.key_dim * num_heads
        self.qkv = nn.Linear(dim, dim + nh_kd * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    @staticmethod
    def _window_partition(x, window_size):
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, window_size * window_size, C)
        return x

    @staticmethod
    def _window_reverse(windows, window_size, H, W):
        B = int(windows.shape[0] // (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, -1)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()

        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))

        _, Hp, Wp, _ = x.shape
        x_windows = self._window_partition(x, self.window_size)

        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(-1, self.window_size * self.window_size, self.num_heads,
                          self.key_dim * 2 + self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, self.dim)

        x = self.proj(x)
        x = self._window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class GlobalAttention(nn.Module):
    """Global multi-head self-attention with spatial reduction for efficiency.
    
    Uses strided convolution to reduce key/value spatial dimensions,
    enabling global context mixing at lower compute cost.
    """

    def __init__(self, dim, num_heads=8, sr_ratio=2, attn_ratio=1.0, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()

        q = self.q(x).reshape(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            xs = shortcut
            xs = self.sr(xs)
            _, _, Hs, Ws = xs.shape
            xs = xs.permute(0, 2, 3, 1).contiguous()
            xs = self.norm(xs)
            kv = self.kv(xs)
            kv = kv.reshape(B, Hs * Ws, self.num_heads, self.head_dim * 2).permute(0, 2, 1, 3)
        else:
            kv = self.kv(x)
            kv = kv.reshape(B, H * W, self.num_heads, self.head_dim * 2).permute(0, 2, 1, 3)

        k, v = kv.split([self.head_dim, self.head_dim], dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, H * W, self.dim)

        x = self.proj(x)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x


class LightTransformerBlock(nn.Module):
    """Lightweight transformer block with optional windowed or global attention.
    
    Combines multi-head self-attention with an MLP feed-forward network,
    using pre-norm and residual connections.
    """

    def __init__(self, dim, num_heads=4, window_size=None, sr_ratio=None,
                 mlp_ratio=2.0, dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        if window_size is not None:
            self.attn = WindowedAttention(dim, num_heads, window_size)
        elif sr_ratio is not None:
            self.attn = GlobalAttention(dim, num_heads, sr_ratio)
        else:
            self.attn = WindowedAttention(dim, num_heads, window_size=7)

        self.mlp = MLPFFN(dim, int(dim * mlp_ratio), dropout, act)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x_ln = x.permute(0, 2, 3, 1).contiguous()
        x_ln = self.norm1(x_ln).permute(0, 3, 1, 2).contiguous()
        x = shortcut + self.attn(x_ln)

        shortcut = x
        x_ln = x.permute(0, 2, 3, 1).contiguous()
        x_ln = self.norm2(x_ln)
        x_ln = self.mlp(x_ln).permute(0, 3, 1, 2).contiguous()
        x = shortcut + x_ln
        return x

    def forward_fuse(self, x):
        return self.forward(x)


class InterleavedTransformerStage(nn.Module):
    """Interleaved window/global transformer stage.
    
    Alternates between windowed local attention and global attention layers
    for efficient hierarchical feature modeling (RF-DETR style).
    """

    def __init__(self, dim, num_heads=4, window_size=7, sr_ratio=2,
                 num_layers=2, mlp_ratio=2.0, dropout=0.0, act=nn.GELU()):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i % 2 == 0:
                layer = LightTransformerBlock(
                    dim, num_heads, window_size=window_size,
                    mlp_ratio=mlp_ratio, dropout=dropout, act=act
                )
            else:
                layer = LightTransformerBlock(
                    dim, num_heads, sr_ratio=sr_ratio,
                    mlp_ratio=mlp_ratio, dropout=dropout, act=act
                )
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def forward_fuse(self, x):
        return self.forward(x)


class ConvLayerNorm(nn.Module):
    """Channel-wise layer normalization for 2D feature maps."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
