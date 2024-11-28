
"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch import Tensor 

import re
import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig, yaml_utils
from src.solver import TASKS

from typing import Dict, List, Optional, Any


__all__ = ["profile_stats"]

def profile_stats(
    model: nn.Module, 
    data: Optional[Tensor]=None, 
    shape: List[int]=[1, 3, 640, 640], 
    verbose: bool=False
) -> Dict[str, Any]:
    
    is_training = model.training

    model.train()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    model.eval()

    if data is None:
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        data = torch.rand(*shape, dtype=dtype, device=device)
        print(device)

    def trace_handler(prof):
        print(prof.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1))

    wait = 0
    warmup = 1
    active = 1
    repeat = 1
    skip_first = 0
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait,
            warmup=warmup,
            active=active,
            repeat=repeat,
            skip_first=skip_first,
        ),
        with_flops=True,
    ) as p:
        n_step = skip_first + (wait + warmup + active) * repeat
        for _ in range(n_step):
            _ = model(data)
            p.step()

    if is_training:
        model.train()
    
    info = p.key_averages().table(sort_by='self_cuda_time_total', row_limit=-1)
    num_flops = sum([float(v.strip()) for v in re.findall('(\d+.?\d+ *\n)', info)]) / active

    if verbose:
        print(info)
        print(f'Total number of trainable parameters: {num_params}')
        print(f'Total number of flops: {int(num_flops)}M with {shape}')

    return {'n_parameters': num_params, 'n_flops': num_flops, 'info': info}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='device',)
    args = parser.parse_args()

    cfg = YAMLConfig(args.config, device=args.device)
    model = cfg.model.to(args.device)

    profile_stats(model, verbose=True)
