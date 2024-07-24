"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch

from ...core import register


__all__ = ['YOLO', ]


@register()
class YOLO(torch.nn.Module):
    __inject__ = ['backbone', 'neck', 'head', ]

    def __init__(self, backbone: torch.nn.Module, neck, head):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head

    def forward(self, x, **kwargs):           
        x = self.backbone(x)
        x = self.neck(x)        
        x = self.head(x)
        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if m is not self and hasattr(m, 'deploy'):
                m.deploy()
        return self 
