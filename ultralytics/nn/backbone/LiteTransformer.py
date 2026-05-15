# LiteTransformer.py
import torch.nn as nn

class LightweightTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads=4, shortcut=True):
        super().__init__()
        # Use c1 for embedding to process input, or a projection to c2
        self.conv_in = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.attn = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(c2, c2 * 2),
            nn.GELU(),
            nn.Linear(c2 * 2, c2)
        )
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x = self.conv_in(x) # Ensure channels match c2 (scaled)
        b, c, h, w = x.shape
        x_flat = x.view(b, c, -1).permute(0, 2, 1) # [B, N, C]
        
        # Self-attention on scaled channels
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = x_flat + attn_out
        x_flat = x_flat + self.ffn(x_flat)
        
        out = x_flat.permute(0, 2, 1).view(b, c, h, w)
        return x + out if self.add else out