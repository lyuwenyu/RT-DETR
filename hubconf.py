"""Copyright(c) 2024 lyuwenyu. All Rights Reserved.
"""


import os
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).absolute().parent / 'rtdetrv2_pytorch'
sys.path.append(str(ROOT))

from src.core import YAMLConfig

import torch
import torch.nn as nn

dependencies = ['torch', 'torchvision',]


def _load_checkpoint(path: str, map_location='cpu'):
    scheme = urlparse(str(path)).scheme
    if not scheme:
        state = torch.load(path, map_location=map_location)
    else:
        state = torch.hub.load_state_dict_from_url(path, map_location=map_location)
    return state


def _build_model(args, ):
    """main
    """
    cfg = YAMLConfig(args.config)

    if args.resume:
        checkpoint = _load_checkpoint(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state
        cfg.model.load_state_dict(state)


    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
            
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    return Model()


CONFIG = {
    # rtdetr
    'rtdetr_r18vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r18vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth',
    },
    'rtdetr_r34vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r34vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth',
    },
    'rtdetr_r50vd_m': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r50vd_m_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth',
    },
    'rtdetr_r50vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r50vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth',
    },
    'rtdetr_r101vd': {
        'config': ROOT / 'configs/rtdetr/rtdetr_r101vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth',
    },

    # rtdetrv2
    'rtdetrv2_r18vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.2/rtdetrv2_r18vd_120e_coco_rerun_48.1.pth',
    },
    'rtdetrv2_r34vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r34vd_120e_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r34vd_120e_coco_ema.pth',
    },
    'rtdetrv2_r50vd_m': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r50vd_m_7x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_m_7x_coco_ema.pth',
    },
    'rtdetrv2_r50vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r50vd_6x_coco_ema.pth',
    },
    'rtdetrv2_r101vd': {
        'config': ROOT / 'configs/rtdetrv2/rtdetrv2_r101vd_6x_coco.yml',
        'resume': 'https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetrv2_r101vd_6x_coco_from_paddle.pth',
    },
}


# rtdetr
def rtdetr_r18vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetr_r18vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r34vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetr_r34vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r50vd_m(pretrained=True):
    args = type('Args', (), CONFIG['rtdetr_r50vd_m'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r50vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetr_r50vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetr_r101vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetr_r101vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


# rtdetrv2
def rtdetrv2_r18vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetrv2_r18vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r34vd(pretrained=True,):
    args = type('Args', (), CONFIG['rtdetrv2_r34vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r50vd_m(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r50vd_m'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r50vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r50vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


def rtdetrv2_r101vd(pretrained=True):
    args = type('Args', (), CONFIG['rtdetrv2_r101vd'])()
    args.resume = args.resume if pretrained else ''
    return _build_model(args, )


rtdetrv2_s = rtdetrv2_r18vd
rtdetrv2_m_r34 = rtdetrv2_r34vd
rtdetrv2_m_r50 = rtdetrv2_r50vd_m
rtdetrv2_l = rtdetrv2_r50vd
rtdetrv2_x = rtdetrv2_r101vd

