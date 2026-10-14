"""Selective Kernel Fusion Module, manuscript section 2.1, equation (1)."""
from torch import nn


class SKFM(nn.Module):
    """Select between two aligned features using per-channel branch weights.

    The manuscript does not specify the MLP bottleneck width. This
    implementation uses max(channels // reduction, min_channels).
    """

    def __init__(self, channels, reduction=8, min_channels=4):
        super().__init__()
        if channels < 1 or reduction < 1 or min_channels < 1:
            raise ValueError("channels, reduction and min_channels must be positive")
        hidden = max(channels // reduction, min_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * channels, 1),
        )

    def forward(self, x, y):
        if x.shape != y.shape:
            raise ValueError("SKFM inputs must have identical NCHW shapes")
        batch, channels, _, _ = x.shape
        logits = self.mlp(self.pool(x + y)).reshape(batch, 2, channels, 1, 1)
        weights = logits.softmax(dim=1)
        return weights[:, 0] * x + weights[:, 1] * y
