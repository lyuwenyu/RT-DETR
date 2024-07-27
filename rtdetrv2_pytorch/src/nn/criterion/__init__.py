"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import torch.nn as nn 
from ...core import register

from .det_criterion import DetCriterion

CrossEntropyLoss = register()(nn.CrossEntropyLoss)
