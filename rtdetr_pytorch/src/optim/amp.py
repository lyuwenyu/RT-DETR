import torch
import torch.nn as nn 
import torch.cuda.amp as amp


from src.core import register
import src.misc.dist as dist 


__all__ = ['GradScaler']

GradScaler = register(amp.grad_scaler.GradScaler)
