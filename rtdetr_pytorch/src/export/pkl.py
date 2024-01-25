from pathlib import Path

import torch
import torch.nn as nn
from src.core import YAMLConfig


class Model(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        print(self.postprocessor.deploy_mode)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        return self.postprocessor(outputs, orig_target_sizes)


def export_torch_pkl(cfg: YAMLConfig, checkpoint_path: Path, output_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "ema" in checkpoint:
        state = checkpoint["ema"]["module"]
    else:
        state = checkpoint["model"]

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    model = Model(cfg)
    torch.save(model, output_path)
