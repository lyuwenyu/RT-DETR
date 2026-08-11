"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
from collections import defaultdict
from packaging import version
from torch import Tensor 

_torchvision_version = version.parse(importlib.metadata.version('torchvision'))

if version.parse('0.16') > _torchvision_version >= version.parse('0.15.2'):
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    from torchvision.transforms.v2.functional import get_spatial_size as get_size
    _boxes_keys = ['format', 'spatial_size']

elif version.parse('0.17') > _torchvision_version >= version.parse('0.16'):
    import torchvision
    torchvision.disable_beta_transforms_warning()

    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.transforms.v2.functional import get_size
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

elif _torchvision_version >= version.parse('0.17'):
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.transforms.v2.functional import get_size
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')



def get_fill(fill_dict, inpt_type):
    # torchvision 0.15.2 keys fill by type in a defaultdict, >= 0.16 falls back to an 'others' key
    if isinstance(fill_dict, defaultdict):
        return fill_dict[inpt_type]
    if inpt_type in fill_dict:
        return fill_dict[inpt_type]
    return fill_dict.get('others', None)


def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    Args:
        tensor (Tensor): input tensor
        key (str): transform to key

    Return:
        Dict[str, TV_Tensor]
    """
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"
    
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size]))
        return BoundingBoxes(tensor, **_kwargs)

    if key == 'masks':
       return Mask(tensor)

