'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 

from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate
from utils import is_by_epoch


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()

        args = self.cfg 
        
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        # best_stat = {'coco_eval_bbox': 0, 'coco_eval_masks': 0, 'epoch': -1, }
        best_stat = {'epoch': -1, }

        start_time = time.time()
        for epoch in range(self.last_epoch + 1, args.epoches):
            if dist.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, self.criterion, self.train_dataloader, self.optimizer, self.device, epoch,
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler,
                lr_warmup=self.lr_warmup, lr_scheduler=self.lr_scheduler)

            if self.lr_scheduler is not None and is_by_epoch(self.lr_scheduler):
                self.lr_scheduler.step()
            
            if epoch % args.checkpoint_step == 0 or epoch == args.epoches - 1:
                if self.output_dir:
                    checkpoint_path = self.output_dir / f'checkpoint{epoch:04}.pth'
                    state_dict = self.state_dict(epoch)
                    if not args.save_optimizer and 'optimizer' in state_dict:
                        state_dict.pop('optimizer')
                    if not args.save_ema and 'ema' in state_dict:
                        state_dict.pop('ema')
                    dist.save_on_master(state_dict, checkpoint_path)

            if epoch % args.val_step == 0 or epoch == args.epoches - 1:
                module = self.ema.module if self.ema else self.model
                test_stats, coco_evaluator = evaluate(
                    module, self.criterion, self.postprocessor, self.val_dataloader, base_ds, self.device, self.output_dir
                )

                # TODO 
                for k in test_stats.keys():
                    if k in best_stat:
                        best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                        best_stat[k] = max(best_stat[k], test_stats[k][0])
                    else:
                        best_stat['epoch'] = epoch
                        best_stat[k] = test_stats[k][0]
                print('best_stat: ', best_stat)
            else:
                test_stats = {}
                coco_evaluator = None

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                # if coco_evaluator is not None:
                #     (self.output_dir / 'eval').mkdir(exist_ok=True)
                #     if "bbox" in coco_evaluator.coco_eval:
                #         filenames = ['latest.pth']
                #         if epoch % 50 == 0:
                #             filenames.append(f'{epoch:03}.pth')
                #         for name in filenames:
                #             torch.save(coco_evaluator.coco_eval["bbox"].eval,
                #                     self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


    def val(self, ):
        self.eval()

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
