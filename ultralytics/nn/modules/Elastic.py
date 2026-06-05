import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelRouter(nn.Module):
    """
    Lightweight policy network that outputs a Gumbel-Softmax distribution 
    to decide whether downstream heavy layers should be skipped.
    """
    def __init__(self, c1, hard=True):
        super().__init__()
        self.hard = hard
        # Downsample features sharply to keep router overhead <1% of total FLOPs
        self.policy_net = nn.Sequential(
            nn.Conv2d(c1, c1 // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c1 // 4),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(c1 // 4, 2)  # Two discrete actions: [0: Skip, 1: Process]
        )
        
    def forward(self, x, tau=1.0):
        logits = self.policy_net(x)
        # Retains differentiable gradients during training
        # Automatically acts as hard selection during model.eval() if hard=True
        return F.gumbel_softmax(logits, tau=tau, hard=self.hard, dim=-1)



class ElasticBlock(nn.Module):
    """
    Elastic block that wraps any native pre-compiled component processed 
    by the modified Ultralytics parser engine.
    """
    def __init__(self, c1: int, heavy_block: nn.Module):
        super().__init__()
        # Ultra-lightweight policy router (<1% FLOP overhead)
        self.router = GumbelRouter(c1=c1, hard=True)
        self.heavy_block = heavy_block
        self.current_action_probs = None
        self.tau = 1.0

    def forward(self, x):
        if not self.training:
            # Accelerated static path evaluation for real-time edge mode
            action_probs = self.router(x)
            if action_probs[0, 1] == 0:  # If index 1 (Route) probability drops to 0
                return x
            return self.heavy_block(x)
            
        action_probs = self.router(x, tau=self.tau)
        self.current_action_probs = action_probs
        
        skip_gate = action_probs[:, 0].view(-1, 1, 1, 1)
        route_gate = action_probs[:, 1].view(-1, 1, 1, 1)
        
        return (skip_gate * x) + (route_gate * self.heavy_block(x))