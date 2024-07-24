"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.cuda.amp as amp

from ..core import register


__all__ = ['GradScaler']

GradScaler = register()(amp.grad_scaler.GradScaler)
