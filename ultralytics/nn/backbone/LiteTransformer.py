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


class DPFATransformerBlock(nn.Module):
    """
    Dual-Path Feature Abstraction (DPFA) Hybrid Block.
    Combines localized CNN textures with global Transformer attention vectors.
    """
    def __init__(self, c1, c2, num_heads=4):
        super().__init__()
        # Ensure the channel counts conform to YAML width-scaling
        self.conv_in = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()
        
        # Path A: Global Attention Context
        self.attn = nn.MultiheadAttention(embed_dim=c2, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(c2, c2 * 2),
            nn.GELU(),
            nn.Linear(c2 * 2, c2)
        )
        
        # Path B: Local Structural Preservation (Depth-wise Conv)
        self.local_path = nn.Conv2d(c2, c2, kernel_size=3, padding=1, groups=c2, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv_in(x)
        b, c, h, w = x.shape
        
        # 1. Extract Global Features via Attention Path
        x_flat = x.view(b, c, -1).permute(0, 2, 1) # [B, H*W, C]
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_global = x_flat + attn_out
        x_global = x_global + self.ffn(x_global)
        x_global = x_global.permute(0, 2, 1).view(b, c, h, w)
        
        # 2. Extract Local Structural Invariances via CNN Path
        x_local = self.act(self.bn(self.local_path(x)))
        
        # 3. Fuse Abstracted Paths (Combines loose recognition + tight boundaries)
        return x_global + x_local