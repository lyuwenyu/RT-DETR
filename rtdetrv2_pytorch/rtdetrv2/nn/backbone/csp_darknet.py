"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import math
import warnings

from .common import get_activation
from ...core import register


def autopad(k, p=None): 
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

def make_divisible(c, d):
    return math.ceil(c / d) * d
    

class Conv(nn.Module):
    def __init__(self, cin, cout, k=1, s=1, p=None, g=1, act='silu') -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, act='silu'):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_, c2, 3, 1, g=g, act=act)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, act='silu'):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c1, c_, 1, 1, act=act)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0, act=act) for _ in range(n)))
        self.cv3 = Conv(2 * c_, c2, 1, act=act)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5, act='silu'):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


@register()
class CSPDarkNet(nn.Module):
    __share__ = ['depth_multi', 'width_multi']

    def __init__(self, in_channels=3, width_multi=1.0, depth_multi=1.0, return_idx=[2, 3, -1], act='silu', ) -> None:
        super().__init__()

        channels = [64, 128, 256, 512, 1024]
        channels = [make_divisible(c * width_multi, 8) for c in channels]

        depths = [3, 6, 9, 3]
        depths = [max(round(d * depth_multi), 1) for d in depths]

        self.layers = nn.ModuleList([Conv(in_channels, channels[0], 6, 2, 2, act=act)])
        for i, (c, d) in enumerate(zip(channels, depths), 1):
            layer = nn.Sequential(*[Conv(c, channels[i], 3, 2, act=act), C3(channels[i], channels[i], n=d, act=act)])
            self.layers.append(layer)

        self.layers.append(SPPF(channels[-1], channels[-1], k=5, act=act))

        self.return_idx = return_idx
        self.out_channels = [channels[i] for i in self.return_idx]
        self.strides = [[2, 4, 8, 16, 32][i] for i in self.return_idx]
        self.depths = depths
        self.act = act

    def forward(self, x):
        outputs = []
        for _, m in enumerate(self.layers):
            x = m(x)
            outputs.append(x)

        return [outputs[i] for i in self.return_idx]


@register()
class CSPPAN(nn.Module):
    """
    P5 ---> 1x1  ---------------------------------> concat --> c3 --> det
             | up                                     | conv /2 
    P4 ---> concat ---> c3 ---> 1x1  -->  concat ---> c3 -----------> det
                                 | up       | conv /2
    P3 -----------------------> concat ---> c3 ---------------------> det
    """
    __share__ = ['depth_multi', ]

    def __init__(self, in_channels=[256, 512, 1024], depth_multi=1., act='silu') -> None:
        super().__init__()
        depth = max(round(3 * depth_multi), 1)

        self.out_channels = in_channels
        self.fpn_stems = nn.ModuleList([Conv(cin, cout, 1, 1, act=act) for cin, cout in zip(in_channels[::-1], in_channels[::-1][1:])])
        self.fpn_csps = nn.ModuleList([C3(cin, cout, depth, False, act=act) for cin, cout in zip(in_channels[::-1], in_channels[::-1][1:])])

        self.pan_stems = nn.ModuleList([Conv(c, c, 3, 2, act=act) for c in in_channels[:-1]])
        self.pan_csps = nn.ModuleList([C3(c, c, depth, False, act=act) for c in in_channels[1:]])

    def forward(self, feats):
        fpn_feats = []
        for i, feat in enumerate(feats[::-1]):
            if i == 0:
                feat = self.fpn_stems[i](feat)
                fpn_feats.append(feat)
            else:
                _feat = F.interpolate(fpn_feats[-1], scale_factor=2, mode='nearest')
                feat = torch.concat([_feat, feat], dim=1)
                feat = self.fpn_csps[i-1](feat)
                if i < len(self.fpn_stems):
                    feat = self.fpn_stems[i](feat)
                fpn_feats.append(feat)

        pan_feats = []
        for i, feat in enumerate(fpn_feats[::-1]):
            if i == 0:
                pan_feats.append(feat)
            else:
                _feat = self.pan_stems[i-1](pan_feats[-1])
                feat = torch.concat([_feat, feat], dim=1)
                feat = self.pan_csps[i-1](feat)
                pan_feats.append(feat)

        return pan_feats


if __name__ == '__main__':

    data = torch.rand(1, 3, 320, 640)

    width_multi = 0.75
    depth_multi = 0.33

    m = CSPDarkNet(3, width_multi=width_multi, depth_multi=depth_multi, act='silu')
    outputs = m(data)
    print([o.shape for o in outputs])

    m = CSPPAN(in_channels=m.out_channels, depth_multi=depth_multi, act='silu')
    outputs = m(outputs)
    print([o.shape for o in outputs])
