"""High-frequency Feature Extraction Module, manuscript section 2.1."""
from torch import nn
from torch.nn import functional as F

if __package__:
    from .skfm import SKFM
else:
    from skfm import SKFM


def conv1x1(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1)


class HFEM(nn.Module):
    """U-shaped decoder with SKFM skip fusion at 1/4 and 1/2 resolution.

    Backbone inputs have 16/24/32 channels. Each upsampled feature and
    its skip feature are projected to equal channels before SKFM fusion.
    Output has 24 channels at approximately 1/4 input resolution.
    """

    def __init__(self):
        super().__init__()
        self.conv_low1 = conv1x1(16, 16)
        self.conv_low2 = conv1x1(24, 32)
        self.conv_low3 = conv1x1(32, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=True)
        self.skfm_quarter = SKFM(32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)
        self.skfm_half = SKFM(16)
        self.conv_adjust = nn.Conv2d(16, 24, 3, stride=2, padding=1)

    def forward(self, low_feature_layer1, low_feature_layer2, low_feature_layer3):
        half = self.conv_low1(low_feature_layer1)
        quarter = self.conv_low2(low_feature_layer2)
        eighth = self.conv_low3(low_feature_layer3)
        up_quarter = self.relu2(self.bn2(self.upconv2(eighth)))
        up_quarter = F.interpolate(up_quarter, size=quarter.shape[-2:],
                                   mode="bilinear", align_corners=True)
        quarter = self.skfm_quarter(up_quarter, quarter)
        up_half = self.relu1(self.bn1(self.upconv1(quarter)))
        up_half = F.interpolate(up_half, size=half.shape[-2:],
                                mode="bilinear", align_corners=True)
        return self.conv_adjust(self.skfm_half(up_half, half))

