import torch
import torch.nn as nn

# -------------------------------------------------
# EdgeGhostBottleneck (c1, c2 required)
# -------------------------------------------------
class EdgeGhostBottleneck(nn.Module):
    def __init__(self, c1, c2, expansion=2.0, stride=1):
        super().__init__()
        hidden = int(c1 * expansion)

        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, hidden, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )

        self.shortcut = (
            nn.Identity() if c1 == c2 and stride == 1 else
            nn.Conv2d(c1, c2, 1, stride, 0, bias=False)
        )

    def forward(self, x):
        return self.conv2(self.dw(self.conv1(x))) + self.shortcut(x)


# -------------------------------------------------
# LiteAttentionFusion (ONLY c1)
# -------------------------------------------------
class LiteAttentionFusion(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(c1, c1, 1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        w = self.act(self.conv(self.pool(x)))
        return x * w


# -------------------------------------------------
# RepDepthwiseBlock (ONLY c1)
# -------------------------------------------------
class RepDepthwiseBlock(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.dw3 = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1, bias=False)
        self.dw5 = nn.Conv2d(c1, c1, 5, 1, 2, groups=c1, bias=False)
        self.pw = nn.Conv2d(c1, c1, 1, 1, 0, bias=False)

        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.dw3(x) + self.dw5(x) + self.pw(x)))
    
class RepMixer(nn.Module):
    """
    FastViT-inspired Reparameterizable Token Mixer[cite: 7].
    Collapses into a single depthwise convolution during inference[cite: 8].
    """
    def __init__(self, c1, c2, k=3, s=1, p=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        if deploy:
            self.reparam_conv = nn.Conv2d(c1, c2, k, s, p, groups=c1)
        else:
            self.dw_conv = nn.Conv2d(c1, c2, k, s, p, groups=c1)
            self.norm = nn.BatchNorm2d(c1)
            self.identity = nn.Identity() if s == 1 and c1 == c2 else None

    def forward(self, x):
        if self.deploy:
            return self.reparam_conv(x)
        out = self.norm(self.dw_conv(x))
        if self.identity:
            out += x
        return out

class LiteAttention(nn.Module):
    """
    Lightweight Global Context Attention for YOLO Neck[cite: 9, 13].
    Captures global dependencies with minimal computational overhead[cite: 10].
    """
    def __init__(self, c1, reduction=16):
        super().__init__()
        self.conv_mask = nn.Conv2d(c1, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.channel_add = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, kernel_size=1),
            nn.LayerNorm([c1 // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // reduction, c1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        mask = self.conv_mask(x).view(b, 1, h * w)
        mask = self.softmax(mask)
        context = torch.matmul(x.view(b, c, h * w), mask.transpose(1, 2))
        context = context.view(b, c, 1, 1)
        return x + self.channel_add(context)