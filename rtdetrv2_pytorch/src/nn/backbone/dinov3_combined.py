"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

DINOv3 组合骨干网络，将 DINOv3 模型和 FPN 适配器组合在一起
"""

import torch
import torch.nn as nn

from ...core import register

from .dinov3_model import DINOv3Model
from .dinov3_adapter import DINOv3FPNAdapter


@register()
class DINOv3Backbone(nn.Module):
    """DINOv3 组合骨干网络

    将 DINOv3 模型和 FPN 适配器组合在一起，输出 RT-DETRv2 所需的多尺度特征:
    - 输入: [B, 3, H, W]
    - 输出: 3 层多尺度特征，每层 256 通道

    输出格式符合 HybridEncoder 的输入要求:
    - 3 个特征图，通道数为 [256, 256, 256]
    - stride 为 [8, 16, 32]
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitb16",
        pretrained_path: str = None,
        layers_to_use: int = 4,
        freeze_backbone: bool = True,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # DINOv3 骨干网络
        self.dinov3 = DINOv3Model(
            name=model_name,
            pretrained_path=pretrained_path,
            layers_to_use=layers_to_use,
            freeze_backbone=freeze_backbone,
        )

        # 获取 DINOv3 的通道数
        embed_dim = self.dinov3.embed_dim  # 768 for ViT-B
        in_channels_list = [embed_dim] * layers_to_use

        # FPN 适配器
        self.adapter = DINOv3FPNAdapter(
            in_channels_list=in_channels_list,
            hidden_dim=hidden_dim,
            out_channels_list=[hidden_dim, hidden_dim, hidden_dim],
            out_strides=[8, 16, 32],
        )

        # 输出配置
        self.strides = [8, 16, 32]
        self.channels = [hidden_dim, hidden_dim, hidden_dim]

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            outputs: 特征列表 List[Tensor], 每层 [B, 256, H/8, H/16, H/32]
        """
        # 获取 DINOv3 多层特征
        feats = self.dinov3(x)

        # 通过 FPN 适配器生成多尺度特征
        outputs = self.adapter(feats)

        return outputs


if __name__ == '__main__':
    # 测试代码
    backbone = DINOv3Backbone(
        model_name='dinov3_vitb16',
        layers_to_use=4,
        freeze_backbone=True,
        hidden_dim=256
    )

    data = torch.rand(1, 3, 640, 640)

    backbone.eval()
    with torch.no_grad():
        outputs = backbone(data)

    for i, output in enumerate(outputs):
        stride = [8, 16, 32][i]
        print(f"Output {i} (stride={stride}): {output.shape}")
