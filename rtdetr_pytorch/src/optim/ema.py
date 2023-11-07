"""
reference: 
https://github.com/ultralytics/yolov5/blob/master/utils/torch_utils.py#L404

by lyuwenyu
"""

import torch
import torch.nn as nn 

import math
from copy import deepcopy



from src.core import register
import src.misc.dist as dist 


__all__ = ['ModelEMA']



@register
class ModelEMA(object):
    """ Model Exponential Moving Average from https://github.com/rwightman/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: nn.Module, decay: float=0.9999, warmups: int=2000):
        super().__init__()

        # Create EMA
        self.module = deepcopy(dist.de_parallel(model)).eval()  # FP32 EMA
        
        # if next(model.parameters()).device.type != 'cpu':
        #     self.module.half()  # FP16 EMA
        
        self.decay = decay 
        self.warmups = warmups
        self.updates = 0  # number of EMA updates
        # self.filter_no_grad = filter_no_grad
        self.decay_fn = lambda x: decay * (1 - math.exp(-x / warmups))  # decay exponential ramp (to help early epochs)
        
        for p in self.module.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay_fn(self.updates)

            msd = dist.de_parallel(model).state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()
            
    def to(self, *args, **kwargs):
        self.module = self.module.to(*args, **kwargs)
        return self

    def update_attr(self, model, include=(), exclude=('process_group', 'reducer')):
        # Update EMA attributes
        self.copy_attr(self.module, model, include, exclude)

    @staticmethod
    def copy_attr(a, b, include=(), exclude=()):
        # Copy attributes from b to a, options to only include [...] and to exclude [...]
        for k, v in b.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(a, k, v)

    def state_dict(self, ):
        return dict(module=self.module.state_dict(), updates=self.updates, warmups=self.warmups)
    
    def load_state_dict(self, state):
        self.module.load_state_dict(state['module']) 
        if 'updates' in state:
            self.updates = state['updates']

    def forwad(self, ):
        raise RuntimeError('ema...')

    def extra_repr(self) -> str:
        return f'decay={self.decay}, warmups={self.warmups}'




class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """
    def __init__(self, model, decay, device="cpu", use_buffers=True):

        self.decay_fn = lambda x: decay * (1 - math.exp(-x / 2000))  
        
        def ema_avg(avg_model_param, model_param, num_averaged):
            decay = self.decay_fn(num_averaged)
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=use_buffers)



