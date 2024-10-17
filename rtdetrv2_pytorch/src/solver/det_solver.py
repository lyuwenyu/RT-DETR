"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""
import os
import time 
import json
import datetime
import wandb

import torch 

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


def save_image_grid(img_list):
    import cv2
    import matplotlib.pyplot as plt
    import math
    # Define grid size (2 rows, 2 columns in this case)
    # Number of images
    num_images = len(img_list)
    
    # Calculate the grid size: rows and columns
    cols = math.ceil(math.sqrt(num_images))  # Number of columns
    rows = math.ceil(num_images / cols)      # Number of rows
    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Flatten the axes array for easy iteration if necessary
    axes = axes.flatten()

    # Plot each image in the grid
    for i in range(num_images):
        axes[i].imshow(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
        axes[i].axis('off')  # Turn off axis for a cleaner look

    # Hide any unused subplots
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.show()

    return fig


class DetSolver(BaseSolver):
    
    def fit(self, ):
        print("Start training")
        self.train()
        args = self.cfg

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1
        
        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)
            
            train_stats = train_one_epoch(
                self.model, 
                self.criterion, 
                self.train_dataloader, 
                self.optimizer, 
                self.device, 
                epoch, 
                max_norm=args.clip_max_norm, 
                print_freq=args.print_freq, 
                ema=self.ema, 
                scaler=self.scaler, 
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                wandb_writer=self.wandb_writer
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()
            
            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

                    if dist_utils.is_main_process():
                        if 'last.pth' not in str(checkpoint_path):
                            at = wandb.Artifact("rtdetr_r18vd", type="model")
                            at.add_file(checkpoint_path)
                            self.wandb_writer.log_artifact(at)

            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module, 
                self.criterion, 
                self.postprocessor, 
                self.val_dataloader, 
                self.evaluator, 
                self.device
            )

            # TODO 
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if self.wandb_writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.wandb_writer.log({f'Test/{k}_{i}'.format(k): v})

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))





    def val(self, ):
        self.eval()
            
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator, batch_results = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)



        import cv2
        thrh = 0.5

        for idx, (samples, _) in enumerate(self.val_dataloader):
            samples = samples.to(self.device)

            outputs = module(samples[:12, :, :, :])
            orig_target_sizes = torch.tensor([[640, 640] for i in range(12)])
            outputs = self.postprocessor.deploy()(outputs, orig_target_sizes)
            _, boxes, scores = outputs

            img_list = []
            for i in range(12):
                scr = scores[i]
                box = boxes[i][scr > thrh]

                img = samples[i, :, :, :].permute(1, 2, 0).numpy()
                img_cv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to cv::Mat

                for i in range(len(box)):
                    cx, cy, w, h = box[i].numpy()
                    # Draw bounding box on the frame
                    cv2.rectangle(img_cv, (int(cx), int(cy)), (int(w), int(h)), (0, 255, 0), 2)

                img_list.append(img_cv)

            fig = save_image_grid(img_list)

            self.wandb_writer.log({f"prediction {idx}": fig})


        from faster_coco_eval import COCO, COCOeval_faster
        from faster_coco_eval.extra import Curves
        from faster_coco_eval.extra import PreviewResults

        cocoGt = COCO("/media/herd-i/Local Disk/herdi/object_detection/dataset/cow/annotations/instances_val.json")
        cocoDt = cocoGt.loadRes(batch_results)
        iouType = "bbox"
        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, extra_calc=True)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        my_table = wandb.Table(columns=list(cocoEval.stats_as_dict.keys()), data=[list(cocoEval.stats_as_dict.values())])
        self.wandb_writer.log({"Evaluation metrics": my_table})

        threshold_iou = 0.5
        cur = Curves(cocoGt, cocoDt, iou_tresh=threshold_iou, iouType=iouType)
        prc_fig = cur.plot_pre_rec(return_fig=True)
        f1_fig = cur.plot_f1_confidence(return_fig=True)
        self.wandb_writer.log({"Precision recall curve": prc_fig})
        self.wandb_writer.log({"F1 curve": f1_fig})


        batch_results = [item for item in batch_results if item['score'] > 0.5]
        cocoDt = cocoGt.loadRes(batch_results)
        iouType = "bbox"
        cocoEval = COCOeval_faster(cocoGt, cocoDt, iouType, extra_calc=True)
        results = PreviewResults(
            cocoGt, cocoDt, iou_tresh=threshold_iou, iouType=iouType, useCats=False
        )
        fig = results.display_matrix(return_fig=True)
        self.wandb_writer.log({"Confusion matrix": fig})

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
