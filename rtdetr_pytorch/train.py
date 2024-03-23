import os
import torch
from src.solver import DetSolver
from src.core import YAMLConfig
from checkpoints import checkpoints
from src.misc.sly_logger import LOGS, Logs


def train(model: str, finetune: bool, config_path: str):

    if finetune:
        checkpoint_url = checkpoints[model]
        name = os.path.basename(checkpoint_url)
        checkpoint_path = f"models/{name}"
        if not os.path.exists(checkpoint_path):
            os.makedirs("models", exist_ok=True)
            torch.hub.download_url_to_file(checkpoint_url, checkpoint_path)
        tuning = checkpoint_path
    else:
        tuning = ''

    cfg = YAMLConfig(
        config_path,
        # resume='',
        tuning=tuning
    )

    solver = DetSolver(cfg)
    solver.fit()

    return cfg


def setup_callbacks():
    def print_iter(logs: Logs):
        print(logs.iter_idx)
        print(logs.loss, logs.lrs, logs.data_time, logs.iter_time, logs.cuda_memory)

    def print_eval(logs: Logs):
        # Charts: AP vs AR (maxDets=100), All APs, All ARs
        print(logs.epoch)
        print(logs.evaluation_metrics)

    LOGS.iter_callback = print_iter
    LOGS.eval_callback = print_eval