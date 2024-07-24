"""
https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.6/ppdet/modeling/backbones/cspresnet.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from collections import OrderedDict

from .common import get_activation

from ...core import register

__all__ = ['CSPResNet']


donwload_url = {
    's': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/CSPResNetb_s_pretrained_from_paddle.pth',
    'm': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/CSPResNetb_m_pretrained_from_paddle.pth',
    'l': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/CSPResNetb_l_pretrained_from_paddle.pth',
    'x': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/CSPResNetb_x_pretrained_from_paddle.pth',
}


class ConvBNLayer(nn.Module):
    def __init__(self, ch_in, ch_out, filter_size=3, stride=1, groups=1, padding=0, act=None):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(ch_out)
        self.act = get_activation(act) 
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', alpha: bool=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBNLayer(
            ch_in, ch_out, 3, stride=1, padding=1, act=None)
        self.conv2 = ConvBNLayer(
            ch_in, ch_out, 1, stride=1, padding=0, act=None)
        self.act = get_activation(act) 

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, ))
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)

        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvBNLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


class BasicBlock(nn.Module):
    def __init__(self,
                 ch_in,
                 ch_out,
                 act='relu',
                 shortcut=True,
                 use_alpha=False):
        super().__init__()
        assert ch_in == ch_out
        self.conv1 = ConvBNLayer(ch_in, ch_out, 3, stride=1, padding=1, act=act)
        self.conv2 = RepVggBlock(ch_out, ch_out, act=act, alpha=use_alpha)
        self.shortcut = shortcut

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        if self.shortcut:
            return x + y
        else:
            return y


class EffectiveSELayer(nn.Module):
    """ Effective Squeeze-Excitation
    From `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
    """

    def __init__(self, channels, act='hardsigmoid'):
        super(EffectiveSELayer, self).__init__()
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.act = get_activation(act)

    def forward(self, x: torch.Tensor):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        x_se = self.act(x_se)
        return x * x_se


class CSPResStage(nn.Module):
    def __init__(self,
                 block_fn,
                 ch_in,
                 ch_out,
                 n,
                 stride,
                 act='relu',
                 attn='eca',
                 use_alpha=False):
        super().__init__()
        ch_mid = (ch_in + ch_out) // 2
        if stride == 2:
            self.conv_down = ConvBNLayer(
                ch_in, ch_mid, 3, stride=2, padding=1, act=act)
        else:
            self.conv_down = None
        self.conv1 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.conv2 = ConvBNLayer(ch_mid, ch_mid // 2, 1, act=act)
        self.blocks = nn.Sequential(*[
            block_fn(
                ch_mid // 2,
                ch_mid // 2,
                act=act,
                shortcut=True,
                use_alpha=use_alpha) for i in range(n)
        ])
        if attn:
            self.attn = EffectiveSELayer(ch_mid, act='hardsigmoid')
        else:
            self.attn = None

        self.conv3 = ConvBNLayer(ch_mid, ch_out, 1, act=act)

    def forward(self, x):
        if self.conv_down is not None:
            x = self.conv_down(x)
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        y = torch.concat([y1, y2], dim=1)
        if self.attn is not None:
            y = self.attn(y)
        y = self.conv3(y)
        return y


@register()
class CSPResNet(nn.Module):
    layers = [3, 6, 6, 3]
    channels = [64, 128, 256, 512, 1024]
    model_cfg = {
        's': {'depth_mult': 0.33, 'width_mult': 0.50, },
        'm': {'depth_mult': 0.67, 'width_mult': 0.75, },
        'l': {'depth_mult': 1.00, 'width_mult': 1.00, },
        'x': {'depth_mult': 1.33, 'width_mult': 1.25, },
    }

    def __init__(self,
                 name: str,
                 act='silu',
                 return_idx=[1, 2, 3],
                 use_large_stem=True,
                 use_alpha=False,
                 pretrained=False):

        super().__init__()        
        depth_mult = self.model_cfg[name]['depth_mult']
        width_mult = self.model_cfg[name]['width_mult']

        channels = [max(round(c * width_mult), 1) for c in self.channels]
        layers = [max(round(l * depth_mult), 1) for l in self.layers]
        act = get_activation(act)

        if use_large_stem:
            self.stem = nn.Sequential(OrderedDict([
                ('conv1', ConvBNLayer(
                    3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                ('conv2', ConvBNLayer(
                    channels[0] // 2,
                    channels[0] // 2,
                    3,
                    stride=1,
                    padding=1,
                    act=act)), ('conv3', ConvBNLayer(
                        channels[0] // 2,
                        channels[0],
                        3,
                        stride=1,
                        padding=1,
                        act=act))]))
        else:
            self.stem = nn.Sequential(OrderedDict([
                ('conv1', ConvBNLayer(
                    3, channels[0] // 2, 3, stride=2, padding=1, act=act)),
                ('conv2', ConvBNLayer(
                    channels[0] // 2,
                    channels[0],
                    3,
                    stride=1,
                    padding=1,
                    act=act))]))

        n = len(channels) - 1
        self.stages = nn.Sequential(OrderedDict([(str(i), CSPResStage(
            BasicBlock,
            channels[i],
            channels[i + 1],
            layers[i],
            2,
            act=act,
            use_alpha=use_alpha)) for i in range(n)]))

        self._out_channels = channels[1:]
        self._out_strides = [4 * 2**i for i in range(n)]
        self.return_idx = return_idx

        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(donwload_url[name], map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            print(f'Load CSPResNet_{name} state_dict')

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        
        return outs
