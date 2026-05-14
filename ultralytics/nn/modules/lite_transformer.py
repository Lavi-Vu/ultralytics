import torch
import torch.nn as nn


class LiteTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)

        hidden_dim = int(dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):

        b, c, h, w = x.shape

        x_flat = x.flatten(2).transpose(1, 2)

        x_norm = self.norm1(x_flat)

        attn_out, _ = self.attn(
            x_norm,
            x_norm,
            x_norm
        )

        x_flat = x_flat + attn_out

        x_flat = x_flat + self.mlp(
            self.norm2(x_flat)
        )

        x = x_flat.transpose(1, 2).reshape(b, c, h, w)

        return x