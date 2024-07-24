"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch 
import torch.nn as nn

from ...core import register


__all__ = ['Classification', 'ClassHead']


@register()
class Classification(torch.nn.Module):
    __inject__ = ['backbone', 'head']

    def __init__(self, backbone: nn.Module, head: nn.Module=None):
        super().__init__()
        
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)

        if self.head is not None:
            x = self.head(x)

        return x 


@register()
class ClassHead(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(hidden_dim, num_classes)  

    def forward(self, x):
        x = x[0] if isinstance(x, (list, tuple)) else x 
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.proj(x)
        return x 
