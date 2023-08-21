"""by lyuwenyu
"""

import torch 
import torch.nn as nn 

from datetime import datetime
from pathlib import Path 

from src.misc import dist
from src.core import BaseConfig


class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        
        self.cfg = cfg 

    def setup(self, ):
        '''Avoid instantiating unnecessary classes 
        '''
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor

        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def train(self, ):
        self.setup()
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler

        # NOTE instantiating order
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.resume(self.cfg.resume)

        self.train_dataloader = dist.warp_loader(self.cfg.train_dataloader, \
            shuffle=self.cfg.train_dataloader.shuffle)
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)


    def eval(self, ):
        self.setup()
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, \
            shuffle=self.cfg.val_dataloader.shuffle)

        if self.cfg.resume:
            print(f'resume from {self.cfg.resume}')
            self.resume(self.cfg.resume)


    def state_dict(self, last_epoch):
        '''state dict
        '''
        state = {}
        state['model'] = dist.de_parallel(self.model).state_dict()
        state['date'] = datetime.now().isoformat()

        # TODO
        state['last_epoch'] = last_epoch

        if self.optimizer is not None:
            state['optimizer'] = self.optimizer.state_dict()

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()
            # state['last_epoch'] = self.lr_scheduler.last_epoch

        if self.ema is not None:
            state['ema'] = self.ema.state_dict()

        if self.scaler is not None:
            state['scaler'] = self.scaler.state_dict()

        return state


    def load_state_dict(self, state):
        '''load state dict
        '''
        # TODO
        if getattr(self, 'last_epoch', None) and 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Loading last_epoch')

        if getattr(self, 'model', None) and 'model' in state:
            if dist.is_parallel(self.model):
                self.model.module.load_state_dict(state['model'])
            else:
                self.model.load_state_dict(state['model'])
            print('Loading model.state_dict')

        if getattr(self, 'ema', None) and 'ema' in state:
            self.ema.load_state_dict(state['ema'])
            print('Loading ema.state_dict')

        if getattr(self, 'optimizer', None) and 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
            print('Loading optimizer.state_dict')

        if getattr(self, 'lr_scheduler', None) and 'lr_scheduler' in state:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])
            print('Loading lr_scheduler.state_dict')

        if getattr(self, 'scaler', None) and 'scaler' in state:
            self.scaler.load_state_dict(state['scaler'])
            print('Loading scaler.state_dict')


    def save(self, path):
        '''save state
        '''
        state = self.state_dict()
        dist.save_on_master(state, path)


    def resume(self, path):
        '''load resume
        '''
        # for cuda:0 memory
        state = torch.load(path, map_location='cpu')
        self.load_state_dict(state)


    def fit(self, ):
        raise NotImplementedError('')


    def val(self, ):
        raise NotImplementedError('')
