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