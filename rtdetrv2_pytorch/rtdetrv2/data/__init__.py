"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from .dataset import *
from .transforms import *
from .dataloader import *

from ._misc import convert_to_tv_tensor




# def set_epoch(self, epoch) -> None:
#     self.epoch = epoch 
# def _set_epoch_func(datasets):
#     """Add `set_epoch` for datasets
#     """
#     from ..core import register
#     for ds in datasets:
#         register(ds)(set_epoch)
# _set_epoch_func([CIFAR10, VOCDetection, CocoDetection])