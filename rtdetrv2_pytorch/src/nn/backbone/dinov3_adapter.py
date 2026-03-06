"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

DINOv3 FPN 适配层，将 DINOv3 特征适配到 RT-DETRv2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core import register


@register()
class DINOv3FPNAdapter(nn.Module):
    """将 DINOv3 特征适配到 RT-DETRv2 的 FPN 适配器

    将多层 ViT 特征转换为多尺度特征:
    - 输入: 4 层 ViT 特征，每层 768 通道
    - 输出: 3 层多尺度特征，每层 256 通道

    ViT 特征是单尺度的 (stride=16)，通过不同采样策略生成多尺度特征。
    """

    def __init__(
        self,
        in_channels_list: list = [768, 768, 768, 768],  # 4 层特征
        hidden_dim: int = 256,
        out_channels_list: list = [256, 256, 256],
        out_strides: list = [8, 16, 32],
    ):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.hidden_dim = hidden_dim
        self.out_strides = out_strides

        # 投影层: 将每层 ViT 特征投影到 hidden_dim
        self.proj_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ) for in_ch in in_channels_list
        ])

        # FPN 融合层
        self.fpn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.GELU()
            ) for _ in range(len(out_channels_list))
        ])

        # 额外的上采样层，用于生成 stride=8 的特征
        self.upsample_layer = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU()
        )

    def forward(self, feats):
        """前向传播

        Args:
            feats: List[Tensor], 4 层 ViT 特征, 每层 [B, 768, H/16, W/16]

        Returns:
            outs: List[Tensor], 3 层多尺度特征, 每层 [B, 256, H/8, H/16, H/32]
        """
        # 先投影到统一维度
        proj_feats = [proj(feat) for proj, feat in zip(self.proj_layers, feats)]

        # proj_feats[0] 对应最浅层 (layer_8), proj_feats[-1] 对应最深层 (layer_11)
        # 我们使用最深的特征作为基础，通过不同采样得到多尺度

        outs = []

        # stride=8: 使用最深层特征上采样 2x
        out = proj_feats[-1]  # [B, 256, H/16, W/16]
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.upsample_layer(out)
        outs.append(out)

        # stride=16: 使用最深层特征
        out = proj_feats[-1]
        out = self.fpn_layers[1](out)
        outs.append(out)

        # stride=32: 使用最浅层特征下采样 2x
        out = proj_feats[0]  # [B, 256, H/16, W/16]
        out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)
        out = self.fpn_layers[2](out)
        outs.append(out)

        return outs


if __name__ == '__main__':
    # 测试代码
    adapter = DINOv3FPNAdapter(
        in_channels_list=[768, 768, 768, 768],
        hidden_dim=256,
        out_channels_list=[256, 256, 256],
        out_strides=[8, 16, 32]
    )

    # 模拟 4 层 ViT 特征
    B, C, H, W = 1, 768, 40, 40  # 640/16=40
    feats = [torch.rand(B, C, H, W) for _ in range(4)]

    adapter.eval()
    with torch.no_grad():
        outs = adapter(feats)

    for i, out in enumerate(outs):
        stride = [8, 16, 32][i]
        print(f"Output {i} (stride={stride}): {out.shape}")
