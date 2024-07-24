"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
from pathlib import Path

import torch 
import torch.nn as nn 

from ..misc import dist_utils
from ._solver import BaseSolver
from .clas_engine import train_one_epoch, evaluate


class ClasSolver(BaseSolver):

    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg 

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params:', n_parameters)

        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        start_time = time.time()
        start_epoch = self.last_epoch + 1
        for epoch in range(start_epoch, args.epoches):

            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(self.model, 
                                        self.criterion, 
                                        self.train_dataloader, 
                                        self.optimizer, 
                                        self.ema, 
                                        epoch=epoch, 
                                        device=self.device)
            self.lr_scheduler.step()
            self.last_epoch += 1

            if output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(epoch), checkpoint_path)

            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(module, self.criterion, self.val_dataloader, self.device)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
            if output_dir and dist_utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


