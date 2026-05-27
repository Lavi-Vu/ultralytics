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
        return self.net(x)

class DynamicTransformerBlock(nn.Module):
    """
    Transformer Block for YOLO26 with Depthwise Separable Attention 
    and Dynamic Token Pruning.
    Integrated for Ultralytics parse_model.
    """
    def __init__(self, c1, c2, *args):
        # c1: input channels (passed by parse_model)
        # c2: output channels (first arg in YAML list)
        # args: optional arguments (e.g., num_heads)
        super().__init__()
        self.c = c2
        self.num_heads = args[0] if len(args) > 0 else 4
        
        # Lazy-initialize controller and qkv in forward (input channels may vary)
        self.controller = None
        self.qkv = None
        self.proj = nn.Linear(c2, c2)
        # we'll apply LayerNorm dynamically in forward based on actual channel dim
        self.norm = None

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        
        # 1. Generate global descriptor internally for RL Controller (GAP)
        global_descriptor = torch.mean(x, dim=(2, 3))

        # Reshape to tokens: [B, N, C]
        flat_x = x.permute(0, 2, 3, 1).flatten(1, 2)  # [B, N, C]

        # Lazy init controller to match actual channel dim
        if self.controller is None:
            self.controller = RLPruningController(input_dim=flat_x.size(-1)).to(x.device)

        # Dynamic LayerNorm to match last-dimension channels
        flat_x = F.layer_norm(flat_x, (flat_x.size(-1),), eps=1e-6)

        # 2. RL-Guided Token Pruning
        if self.controller is not None:
            # Predict pruning ratio rho
            rho = self.controller(global_descriptor) # [B, 1]
            
            # Calculate tokens to keep
            k = int(N * (1 - rho.detach().mean().item()))
            k = max(1, min(k, N))
            
            # Importance Scoring (L2 Norm)
            scores = torch.norm(flat_x, p=2, dim=-1) # [B, N]
            _, indices = torch.topk(scores, k, dim=-1)
            
            batch_indices = torch.arange(B).view(-1, 1).expand(B, k).to(x.device)
            flat_x = flat_x[batch_indices, indices] # [B, k, C]
            current_k = k
        else:
            current_k = N

        # 3. Efficient MHA
        # Lazy-init qkv to accept the actual token/channel dimension
        if self.qkv is None or self.qkv.in_features != flat_x.size(-1):
            self.qkv = nn.Linear(flat_x.size(-1), self.c * 3, bias=False).to(x.device)

        # compute per-head dimension and validate
        if self.c % self.num_heads != 0:
            raise ValueError(f"channels {self.c} not divisible by num_heads {self.num_heads}")
        d = self.c // self.num_heads

        qkv = self.qkv(flat_x).view(B, current_k, 3, self.num_heads, d).permute(2, 0, 3, 1, 4)
        q, k_tensor, v = qkv[0], qkv[1], qkv[2]  # q,k,v -> [B, heads, k, d]

        attn = (q @ k_tensor.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, current_k, self.c)
        out = self.proj(out)

        # 4. Restore Spatial Dimensions
        if current_k < N:
            out = out.mean(dim=1, keepdim=True).expand(B, N, self.c)
        
        out = out.transpose(1, 2).reshape(B, self.c, H, W)
        return out + x
