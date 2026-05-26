import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv

class RLPruningController(nn.Module):
    """
    A lightweight RL policy network that predicts the pruning ratio 
    based on a global image descriptor.
    """
    def __init__(self, input_dim=256, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # Outputs ratio rho between 0 and 1
        )

    def forward(self, x):
        # x is a global descriptor (e.g., GAP of the image)
        return self.net(x)

class DynamicTransformerBlock(nn.Module):
    """
    Transformer Block for YOLO26 with Depthwise Separable Attention 
    and Dynamic Token Pruning.
    """
    def __init__(self, c1, c2, num_heads=4, pruning_controller=None):
        super().__init__()
        self.c = c2
        self.num_heads = num_heads
        self.controller = pruning_controller
        
        # Depthwise Separable Attention to keep it lightweight
        self.qkv = nn.Linear(c1, c2 * 3, bias=False)
        self.proj = nn.Linear(c2, c2)
        self.norm = nn.LayerNorm(c2)

    def forward(self, x, global_descriptor=None):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        
        # Reshape to tokens: [B, N, C]
        flat_x = x.permute(0, 2, 3, 1).flatten(1, 2) 
        flat_x = self.norm(flat_x)

        # 1. RL-Guided Token Pruning
        if self.controller is not None and global_descriptor is not None:
            # Get pruning ratio rho from RL Controller
            rho = self.controller(global_descriptor) # [B, 1]
            
            # Calculate number of tokens to keep
            k = int(N * (1 - rho.detach().mean().item()))
            k = max(1, min(k, N))
            
            # Simple Importance Scoring: Use L2 norm of tokens as a proxy for importance
            scores = torch.norm(flat_x, p=2, dim=-1) # [B, N]
            _, indices = torch.topk(scores, k, dim=-1)
            
            # Gather the top-k tokens
            batch_indices = torch.arange(B).view(-1, 1).expand(B, k).to(x.device)
            flat_x = flat_x[batch_indices, indices] # [B, k, C]
            current_k = k
        else:
            current_k = N

        # 2. Efficient Multi-Head Attention (MHA)
        qkv = self.qkv(flat_x).reshape(B, current_k, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, heads, k, d]

        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1]**-0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, current_k, self.c)
        out = self.proj(out)

        # 3. Restore Spatial Dimensions (Upsample if pruned)
        if current_k < N:
            out = out.mean(dim=1, keepdim=True).expand(B, N, self.c)
        
        out = out.transpose(1, 2).reshape(B, self.c, H, W)
        return out + x # Residual connection
