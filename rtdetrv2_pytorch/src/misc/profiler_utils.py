"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import re
import torch
import torch.nn as nn
from torch import Tensor 

from typing import List

def stats(
    model: nn.Module, 
    data: Tensor=None, 
    input_shape: List=[1, 3, 640, 640], 
    device: str='cpu', 
    verbose=False) -> str:
    
    is_training = model.training

    model.train()
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    model.eval()
    model = model.to(device)

    if data is None:
        data = torch.rand(*input_shape, device=device)
        
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    num_active = 2
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=num_active,
            repeat=1
        ),
        # on_trace_ready=trace_handler,
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # with_modules=True,
        with_flops=True,
    ) as p:
        for _ in range(5):
            _ = model(data)
            p.step()

    if is_training:
        model.train()
    
    info = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
    num_flops = sum([float(v.strip()) for v in re.findall('(\d+.?\d+ *\n)', info)]) / num_active

    if verbose:
        # print(info)
        print(f'Total number of trainable parameters: {num_params}')
        print(f'Total number of flops: {int(num_flops)}M with {input_shape}')

    return {'n_parameters': num_params, 'n_flops': num_flops, 'info': info}
