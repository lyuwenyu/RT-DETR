"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch import Tensor
from typing import List, Tuple

from .common import FrozenBatchNorm2d
from ...core import register


__all__ = ['HGNetv2']


class LearnableAffineBlock(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor([scale_value]))
        self.bias = nn.Parameter(torch.tensor([bias_value]))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x + self.bias


class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 groups=1,
                 use_act=True,
                 use_lab=False):
        super().__init__()
        self.use_act = use_act
        self.use_lab = use_lab
        if padding == 'same':
            self.conv = nn.Sequential(
                nn.ZeroPad2d([0, 1, 0, 1]),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    groups=groups,
                    bias=False
                )
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding=(kernel_size - 1) // 2,
                groups=groups,
                bias=False
            )
        self.bn = nn.BatchNorm2d(out_channels)
        if self.use_act:
            self.act = nn.ReLU()
            if self.use_lab:
                self.lab = LearnableAffineBlock()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        if self.use_act:
            x = self.act(x)
            if self.use_lab:
                x = self.lab(x)
        return x


class LightConvBNAct(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 use_lab=False):
        super().__init__()
        self.conv1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            use_act=False,
            use_lab=use_lab
        )
        self.conv2 = ConvBNAct(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=out_channels,
            use_act=True,
            use_lab=use_lab
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StemBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 use_lab=False):
        super().__init__()
        self.stem1 = ConvBNAct(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab
        )
        self.stem2a = ConvBNAct(
            in_channels=mid_channels,
            out_channels=mid_channels // 2,
            kernel_size=2,
            stride=1,
            padding='same',
            use_lab=use_lab
        )
        self.stem2b = ConvBNAct(
            in_channels=mid_channels // 2,
            out_channels=mid_channels,
            kernel_size=2,
            stride=1,
            padding='same',
            use_lab=use_lab
        )
        self.stem3 = ConvBNAct(
            in_channels=mid_channels * 2,
            out_channels=mid_channels,
            kernel_size=3,
            stride=2,
            use_lab=use_lab
        )
        self.stem4 = ConvBNAct(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab
        )

        self.pool = nn.Sequential(
            nn.ZeroPad2d([0, 1, 0, 1]),
            nn.MaxPool2d(2, 1, ceil_mode=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem1(x)
        x2 = self.stem2a(x)
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.concat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)

        return x


class HG_Block(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3,
                 layer_num=6,
                 identity=False,
                 light_block=True,
                 use_lab=False):
        super().__init__()
        self.identity = identity

        self.layers = nn.ModuleList()
        block_type = "LightConvBNAct" if light_block else "ConvBNAct"
        for i in range(layer_num):
            self.layers.append(
                eval(block_type)(in_channels=in_channels
                                 if i == 0 else mid_channels,
                                 out_channels=mid_channels,
                                 stride=1,
                                 kernel_size=kernel_size,
                                 use_lab=use_lab))
        # feature aggregation
        total_channels = in_channels + layer_num * mid_channels
        self.aggregation_squeeze_conv = ConvBNAct(
            in_channels=total_channels,
            out_channels=out_channels // 2,
            kernel_size=1,
            stride=1,
            use_lab=use_lab)
        self.aggregation_excitation_conv = ConvBNAct(
            in_channels=out_channels // 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            use_lab=use_lab)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.concat(output, dim=1)
        x = self.aggregation_squeeze_conv(x)
        x = self.aggregation_excitation_conv(x)
        if self.identity:
            x = x + identity
        return x


class HG_Stage(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 block_num,
                 layer_num=6,
                 downsample=True,
                 light_block=True,
                 kernel_size=3,
                 use_lab=False):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.downsample = ConvBNAct(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                stride=2,
                groups=in_channels,
                use_act=False,
                use_lab=use_lab)

        blocks_list = []
        for i in range(block_num):
            blocks_list.append(
                HG_Block(
                    in_channels=in_channels if i == 0 else out_channels,
                    mid_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    layer_num=layer_num,
                    identity=False if i == 0 else True,
                    light_block=light_block,
                    use_lab=use_lab))
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        if self.downsample:
            x = self.downsample(x)
        x = self.blocks(x)
        return x


@register()
class HGNetv2(nn.Module):
    """
    Args:
        stem_channels: list. Number of channels for the stem block.
        stage_type: str. The stage configuration of PPHGNet. such as the number of channels, stride, etc.
        use_lab: boolean. Whether to use LearnableAffineBlock in network.
        lr_mult_list: list. Control the learning rate of different stages.
    Returns:
        model: nn.Module.
    """

    arch_configs = {
        'L': {
            'stem_channels': [3, 32, 48],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [48, 48, 128, 1, False, False, 3, 6],
                "stage2": [128, 96, 512, 1, True, False, 3, 6],
                "stage3": [512, 192, 1024, 3, True, True, 5, 6],
                "stage4": [1024, 384, 2048, 1, True, True, 5, 6],
            },
            'url': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_L_ssld_pretrained_from_paddle.pth',

        },
        'X': {
            'stem_channels': [3, 32, 64],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [64, 64, 128, 1, False, False, 3, 6],
                "stage2": [128, 128, 512, 2, True, False, 3, 6],
                "stage3": [512, 256, 1024, 5, True, True, 5, 6],
                "stage4": [1024, 512, 2048, 2, True, True, 5, 6],
            },
            'url': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_X_ssld_pretrained_from_paddle.pth',

        },
        'H': {
            'stem_channels': [3, 48, 96],
            'stage_config': {
                # in_channels, mid_channels, out_channels, num_blocks, downsample, light_block, kernel_size, layer_num
                "stage1": [96, 96, 192, 2, False, False, 3, 6],
                "stage2": [192, 192, 512, 3, True, False, 3, 6],
                "stage3": [512, 384, 1024, 6, True, True, 5, 6],
                "stage4": [1024, 768, 2048, 3, True, True, 5, 6],
            },
            'url': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/PPHGNetV2_H_ssld_pretrained_from_paddle.pth',
        }
    }

    def __init__(self,
                 name,
                 use_lab=False,
                 return_idx=[1, 2, 3],
                 freeze_at=-1,
                 freeze_norm=False,
                 pretrained=False):
        super().__init__()
        self.use_lab = use_lab
        self.return_idx = return_idx

        stem_channels = self.arch_configs[name]['stem_channels']
        stage_config = self.arch_configs[name]['stage_config']
        download_url = self.arch_configs[name]['url']

        self._out_strides = [4, 8, 16, 32]
        self._out_channels = [stage_config[k][2] for k in stage_config]

        # stem
        self.stem = StemBlock(
            in_channels=stem_channels[0],
            mid_channels=stem_channels[1],
            out_channels=stem_channels[2],
            use_lab=use_lab
        )

        # stages
        self.stages = nn.ModuleList()
        for i, k in enumerate(stage_config):
            in_channels, mid_channels, out_channels, block_num, downsample, light_block, kernel_size, layer_num = stage_config[
                k]
            self.stages.append(
                HG_Stage(
                    in_channels,
                    mid_channels,
                    out_channels,
                    block_num,
                    layer_num,
                    downsample,
                    light_block,
                    kernel_size,
                    use_lab))

        self._init_weights()

        if freeze_at >= 0:
            self._freeze_parameters(self.stem)
            for i in range(min(freeze_at, 4)):
                self._freeze_parameters(self.stages[i])

        if freeze_norm:
            self._freeze_norm(self)

        if pretrained:
            if isinstance(pretrained, bool) or 'http' in pretrained:
                state = torch.hub.load_state_dict_from_url(download_url, map_location='cpu')
            else:
                state = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(state)
            print(f'Load HGNetv2_{name} state_dict')
        

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d)):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.constant_(m.bias, 0)

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.BatchNorm2d):
            m = FrozenBatchNorm2d(m.num_features)
        else:
            for name, child in m.named_children():
                _child = self._freeze_norm(child)
                if _child is not child:
                    setattr(m, name, _child)
        return m


    def forward(self, x: Tensor) -> List[Tensor]:
        x = self.stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.return_idx:
                outs.append(x)
        return outs



if __name__ == '__main__':

    m = HGNetv2(name='X', pretrained=False, freeze_at=-1, freeze_norm=False)
    data = torch.randn(1, 3, 640, 640)

    output = m(data)
    print([o.shape for o in output])

    output[0].mean().backward()
