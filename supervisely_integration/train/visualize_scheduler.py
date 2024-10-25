import torch
from torch.optim.lr_scheduler import LRScheduler

from rtdetr_pytorch.utils import is_by_epoch


def test_schedulers(
    lr_scheduler: LRScheduler, lr_warmup: LRScheduler, dummy_optim, dataloader_len, total_epochs
):
    lrs = []
    for ep in range(total_epochs):
        for i in range(dataloader_len):
            lrs.append(dummy_optim.param_groups[0]["lr"])
            if lr_warmup is not None:
                lr_warmup.step()
            if lr_scheduler is not None and not is_by_epoch(lr_scheduler):
                lr_scheduler.step()
        if lr_scheduler is not None and is_by_epoch(lr_scheduler):
            lr_scheduler.step()
    x = list(range(len(lrs)))
    return x, lrs
