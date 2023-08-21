""""by lyuwenyu
"""


import torch 
import torch.nn as nn 

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG


__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
SanitizeBoundingBox = register(T.SanitizeBoundingBox)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)



@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                    # op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)


@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt

