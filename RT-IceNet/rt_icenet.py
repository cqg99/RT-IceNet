"""RT-IceNet for four-class sea-ice semantic segmentation."""
from torch import nn
from torch.nn import functional as F

if __package__:
    from .backbone import MobileNetV2Backbone
    from .hfem import HFEM
    from .siem import SIEM
    from .pafm import PAFM
else:
    from backbone import MobileNetV2Backbone
    from hfem import HFEM
    from siem import SIEM
    from pafm import PAFM


class RTIceNet(nn.Module):
    """MobileNetV2 backbone with HFEM, SIEM and PAFM streams.

    The default output stride is eight, matching the manuscript. Output
    contains unnormalized class logits at the original input resolution.
    Pretrained=True downloads MobileNetV2 weights if not cached locally.
    """

    def __init__(self, num_classes=4, backbone="mobilenet", pretrained=True,
                 downsample_factor=8):
        super().__init__()
        if backbone != "mobilenet":
            raise ValueError("RT-IceNet uses the MobileNetV2 backbone; select 'mobilenet'")
        if downsample_factor not in (8, 16):
            raise ValueError("downsample_factor must be 8 or 16")
        self.backbone = MobileNetV2Backbone(downsample_factor, pretrained)
        self.hfem = HFEM()
        self.siem = SIEM(320, branch_channels=96, out_channels=256)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(24, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.pafm = PAFM(256)
        self.cls_conv = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        output_size = x.shape[-2:]
        low1, low2, low3, deep = self.backbone(x)
        high_frequency = self.shortcut_conv(self.hfem(low1, low2, low3))
        semantics = self.siem(deep)
        semantics = F.interpolate(semantics, size=high_frequency.shape[-2:],
                                  mode="bilinear", align_corners=True)
        logits = self.cls_conv(self.pafm(semantics, high_frequency))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=True)
