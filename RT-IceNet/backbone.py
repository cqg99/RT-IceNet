"""MobileNetV2-derived multi-scale backbone for RT-IceNet."""
import torch.nn as nn

if __package__:
    from .mobilenetv2 import mobilenetv2
else:
    from mobilenetv2 import mobilenetv2


class MobileNetV2Backbone(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        from functools import partial
        
        model           = mobilenetv2(pretrained)
        self.features   = model.features[:-1]

        self.total_idx  = len(self.features)
        self.down_idx   = [2, 4, 7, 14]

        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=4)
                )
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(
                    partial(self._nostride_dilate, dilate=2)
                )
        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        low_feature_layer1 = self.features[:self.down_idx[0]](x)
        x = self.features[self.down_idx[0]:self.down_idx[1]](low_feature_layer1)
        low_feature_layer2 = x

        x = self.features[self.down_idx[1]:self.down_idx[2]](x)
        low_feature_layer3 = x

        x = self.features[self.down_idx[2]:](x)
        
        return low_feature_layer1, low_feature_layer2, low_feature_layer3, x
    

