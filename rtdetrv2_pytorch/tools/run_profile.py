"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
import os
import sys

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from typing import Any, Dict, List, Optional

from src.core import YAMLConfig, yaml_utils

__all__ = ["profile_stats"]

def _auto_scale_flops(flops: float):
    """Copied from torch.profiler.profile"""
    flop_headers = [
        "",
        "K",
        "M",
        "G",
        "T",
        "P",
    ]
    assert flops > 0
    log_flops = max(0, min(math.log10(flops) / 3, float(len(flop_headers) - 1)))
    assert log_flops >= 0 and log_flops < len(flop_headers)
    return (pow(10, (math.floor(log_flops) * -3.0)), flop_headers[int(log_flops)])

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

    statistics = p.key_averages()
    info = statistics.table(sort_by='self_cuda_time_total', row_limit=-1)
    num_flops = sum(event.flops for event in statistics if event.flops > 0) / active
    (flops_scale, flops_header) = _auto_scale_flops(num_flops)

    if verbose:
        print(info)
        print(f'Total number of trainable parameters: {num_params}')
        print(f'Total number of flops: {num_flops * flops_scale:.3f}{flops_header} with {shape}')

    return {'n_parameters': num_params, 'n_flops': num_flops, 'info': info}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='device',)
    parser.add_argument('-u', '--update', nargs='+', help='Update yaml config from command line.')
    args = parser.parse_args()

    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)
    model = cfg.model.to(args.device)

    profile_stats(model, verbose=True)
