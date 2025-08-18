"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 

from ..misc import (MetricLogger, SmoothedValue, reduce_dict)


def train_one_epoch(model: nn.Module, criterion: nn.Module, dataloader, optimizer, ema, epoch, device):
    """
    """
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    print_freq = 100
    header = 'Epoch: [{}]'.format(epoch)

    for imgs, labels in metric_logger.log_every(dataloader, print_freq, header):
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        loss: torch.Tensor = criterion(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if ema is not None:
            ema.update(model)

        loss_reduced_values = {k: v.item() for k, v in reduce_dict({'loss': loss}).items()}
        metric_logger.update(**loss_reduced_values)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats



@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()

    metric_logger = MetricLogger(delimiter="  ")
    # metric_logger.add_meter('acc', SmoothedValue(window_size=1, fmt='{global_avg:.4f}'))
    # metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('acc', SmoothedValue(window_size=1))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1))

    header = 'Test:'
    for imgs, labels in metric_logger.log_every(dataloader, 10, header):
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs)

        acc = (preds.argmax(dim=-1) == labels).sum() / preds.shape[0]
        loss = criterion(preds, labels)

        dict_reduced = reduce_dict({'acc': acc, 'loss': loss})
        reduced_values = {k: v.item() for k, v in dict_reduced.items()}
        metric_logger.update(**reduced_values)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


