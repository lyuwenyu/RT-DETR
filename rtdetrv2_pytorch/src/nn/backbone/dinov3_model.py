"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.

DINOv3 骨干网络封装，用于 RT-DETRv2
"""

import sys
import os
import torch
import torch.nn as nn

# 添加 DINOv3 项目路径
dinov3_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), '..', '..', 'dinov3')
if dinov3_path not in sys.path:
    sys.path.insert(0, dinov3_path)

from ...core import register


@register()
class DINOv3Model(nn.Module):
    """DINOv3 骨干网络封装，用于 RT-DETRv2

    将 DINOv3 ViT 模型的输出转换为检测器所需的格式:
    - 获取骨干网络的多层特征 (默认最后4层)
    - 输出多尺度特征列表

    输出格式: List[Tensor], 每个 Tensor 形状为 [B, C, H, W]
    """

    def __init__(
        self,
        name: str = "dinov3_vitb16",
        pretrained_path: str = None,
        layers_to_use: int = 4,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.name = name
        self.layers_to_use = layers_to_use

        # 加载 DINOv3 模型
        if name == "dinov3_vitb16":
            from dinov3.hub import backbones as dinov3_backbones
            self.backbone = dinov3_backbones.dinov3_vitb16(pretrained=True)
        elif name == "dinov3_vitl16":
            from dinov3.hub import backbones as dinov3_backbones
            self.backbone = dinov3_backbones.dinov3_vitl16(pretrained=True)
        elif name == "dinov3_vits16":
            from dinov3.hub import backbones as dinov3_backbones
            self.backbone = dinov3_backbones.dinov3_vits16(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {name}")

        self.patch_size = self.backbone.patch_size  # 16

        # 冻结骨干网络
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 获取模型的 embed_dim
        self.embed_dim = self.backbone.embed_dim  # 768 for ViT-B

        # 设置输出通道和 stride
        # ViT 是单尺度特征，所有层的 stride 都是 patch_size
        self.strides = [self.patch_size] * layers_to_use
        self.channels = [self.embed_dim] * layers_to_use

    def forward(self, x: torch.Tensor):
        """前向传播

        Args:
            x: 输入图像 [B, 3, H, W]

        Returns:
            outputs: 特征列表 List[Tensor], 每个 [B, embed_dim, H/patch_size, W/patch_size]
        """
        # 获取多层特征
        # get_intermediate_layers 返回 layers_to_use 个特征，每个形状为 [B, embed_dim, H', W']
        xs = self.backbone.get_intermediate_layers(
            x, n=self.layers_to_use, reshape=True
        )

        # 返回特征列表
        return list(xs)


if __name__ == '__main__':
    # 测试代码
    model = DINOv3Model(name='dinov3_vitb16', layers_to_use=4, freeze_backbone=True)
    data = torch.rand(1, 3, 640, 640)

    # 设置为 eval 模式以禁用 dropout 等
    model.eval()

    with torch.no_grad():
        outputs = model(data)

    for i, output in enumerate(outputs):
        print(f"Layer {i}: {output.shape}")
