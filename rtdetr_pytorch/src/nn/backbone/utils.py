"""
https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py

by lyuwenyu
"""

from collections import OrderedDict
from typing import Dict, List


import torch.nn as nn 


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    """

    _version = 3

    def __init__(self, model: nn.Module, return_layers: List[str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model. {}"\
                .format([name for name, _ in model.named_children()]))
        orig_return_layers = return_layers
        return_layers = {str(k): str(k)  for k in return_layers}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        # out = OrderedDict()
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                # out_name = self.return_layers[name]
                # out[out_name] = x
                outputs.append(x)
        
        return outputs

