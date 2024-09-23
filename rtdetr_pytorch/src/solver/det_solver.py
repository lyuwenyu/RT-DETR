'''
by lyuwenyu
'''
import time 
import json
import datetime

import torch 
import numpy as np
import os
import copy
import cv2
from src.misc import dist
from src.data import get_coco_api_from_dataset

from .solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


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
                args.clip_max_norm, print_freq=args.log_step, ema=self.ema, scaler=self.scaler)

            self.lr_scheduler.step()
            
            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_step == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist.save_on_master(self.state_dict(epoch), checkpoint_path)

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


            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if self.output_dir and dist.is_main_process():
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

        base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, base_ds, self.device, self.output_dir)
                
        if self.output_dir:
            dist.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")
        
        return
        
    def infer(self,):
        self.eval()

        # base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
        model = self.ema.module if self.ema else self.model
        checkpoint = torch.load('/home/multi-gpu/RT-DETR_regnet_dla_support/rtdetr_pytorch/output/rtdetr_regnet_6x_coco/checkpoint0071.pth')
        
        model.load_state_dict(checkpoint['model'])
        model = model.to(device='cuda')
        model.eval()
        postprocessor = self.postprocessor
        # data_loader =  self.val_dataloader

        path = '/home/multi-gpu/RT-DETR_regnet_dla_support/rtdetr_pytorch/dataset/val2017/000000000139.jpg'
        os.makedirs('infer',exist_ok=True)
        path_infer = '/home/multi-gpu/RT-DETR_regnet_dla_support/rtdetr_pytorch/infer'
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
        # out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (1920, 1080))
        # for images in tqdm(sorted(os.listdir(path)),desc = "Inference"):
        img = cv2.imread(os.path.join(path))
        resized_image = cv2.resize(img, (640, 640))

        resized_image_float = resized_image.astype(np.float32)
        # Normalize the image
        mean = np.array([0.0, 0.0, 0.0])
        std = np.array([255.0, 255.0, 255.0])
        #obj image
        normalized_image = (resized_image_float - mean) / std
        normalized_image = normalized_image.transpose(2,0,1)
        normalized_image = torch.tensor(normalized_image,dtype= torch.float32).to(device= 'cuda')
        normalized_image = normalized_image.unsqueeze(0)
        
        
        start = time.time()
        outputs = model(normalized_image)        
        end = time.time()
        print("Model Inference Time: ", end - start)

       
        # import pdb; pdb.set_trace()


        # dst = img
        dst = copy.deepcopy(img)
        
        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)        

        orig_target_sizes = torch.tensor([1920,1080]).to(device='cuda')
        results = postprocessor(outputs, orig_target_sizes)
        
        labels = results[0]['labels']
        boxes = results[0]['boxes']
        scores = results[0]['scores']
        score_threshold = 0.4
        for label, box, score in zip(labels, boxes, scores):
                    
            if score > score_threshold:
                xmin,ymin, xmax, ymax = box.tolist()
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

                    # Draw the bounding box
            
                cv2.rectangle(dst, (xmin, ymin), (xmax, ymax), (0, 255, 255), 5)

            # Annotate with the class name and score
                label_text = f"Class {label}, Score {score:.2f}"
                cv2.putText(dst, label_text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


        cv2.imwrite(path_infer + 'hm.jpg',dst)
        # path_infer = 
        # path_infer = '/home/osama/factory-copilot/rtdetr_pytorch/infer/0000.jpg'
        # cv2.imwrite(path_infer, dst)  # Save the image  
        # out.write(dst)
        # import pdb; pdb.set_trace()

        # out.release()
            

