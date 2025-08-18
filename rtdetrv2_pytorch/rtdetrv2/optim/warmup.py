"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from torch.optim.lr_scheduler import LRScheduler

from ..core import register


class Warmup(object):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int=-1) -> None:
        self.lr_scheduler = lr_scheduler
        self.warmup_end_values = [pg['lr'] for pg in lr_scheduler.optimizer.param_groups]
        self.last_step = last_step
        self.warmup_duration = warmup_duration
        self.step()

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != 'lr_scheduler'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_warmup_factor(self, step, **kwargs):
        raise NotImplementedError

    def step(self, ):
        self.last_step += 1
        if self.last_step >= self.warmup_duration:
            return
        factor = self.get_warmup_factor(self.last_step)
        for i, pg in enumerate(self.lr_scheduler.optimizer.param_groups):
            pg['lr'] = factor * self.warmup_end_values[i]
    
    def finished(self, ):
        if self.last_step >= self.warmup_duration:
            return True 
        return False


@register()
class LinearWarmup(Warmup):
    def __init__(self, lr_scheduler: LRScheduler, warmup_duration: int, last_step: int = -1) -> None:
        super().__init__(lr_scheduler, warmup_duration, last_step)

    def get_warmup_factor(self, step):
        return min(1.0, (step + 1) / self.warmup_duration)

