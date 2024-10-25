import os
from functools import partial
from typing import Callable, Optional
from urllib.request import urlopen

import torch
from checkpoints import checkpoints
from src.core import YAMLConfig
from src.misc.sly_logger import LOGS, Logs
from src.solver import DetSolver

import supervisely as sly
import supervisely_integration.train.globals as g
from supervisely.app.widgets import Button, Field, Progress


def train(
    model: str,
    finetune: bool,
    config_path: str,
    progress_download_model: Progress,
    progress_bar_epochs: Progress,
    progress_bar_iters: Progress,
    stop_button: Button,
    charts_grid: Field,
):
    def download_monitor(monitor, api: sly.Api, progress: sly.Progress):
        value = monitor
        if progress.total == 0:
            progress.set(value, monitor.len, report=False)
        else:
            progress.set_current_value(value, report=False)
        weights_pbar.update(progress.current)

    if finetune:
        if g.model_mode == g.MODEL_MODES[0]:
            checkpoint_url = checkpoints[model]
            name = os.path.basename(checkpoint_url)
            checkpoint_path = f"models/{name}"
            if not os.path.exists(checkpoint_path):
                os.makedirs("models", exist_ok=True)
                # torch.hub.download_url_to_file(checkpoint_url, checkpoint_path)
                with urlopen(checkpoint_url) as file:
                    weights_size = file.length

                progress = sly.Progress(
                    message="",
                    total_cnt=weights_size,
                    is_size=True,
                )
                progress_cb = partial(download_monitor, api=g.api, progress=progress)
                with progress_download_model(
                    message="Downloading model weights...",
                    total=weights_size,
                    unit="bytes",
                    unit_scale=True,
                ) as weights_pbar:
                    sly.fs.download(
                        url=checkpoint_url,
                        save_path=checkpoint_path,
                        progress=progress_cb,
                    )
            tuning = checkpoint_path
        else:
            checkpoint_url = model
            name = os.path.basename(checkpoint_url)
            checkpoint_path = f"models/{name}"
            checkpoint_info = g.api.file.get_info_by_path(g.TEAM_ID, checkpoint_url)
            weights_size = checkpoint_info.sizeb
            with progress_download_model(
                message="Downloading model weights...",
                total=weights_size,
                unit="bytes",
                unit_scale=True,
            ) as weights_pbar:
                if not os.path.exists(checkpoint_path):
                    os.makedirs("models", exist_ok=True)
                    g.api.file.download(
                        g.TEAM_ID, checkpoint_url, checkpoint_path, progress_cb=weights_pbar.update
                    )
            tuning = checkpoint_path
    else:
        tuning = ""

    cfg = YAMLConfig(
        config_path,
        # resume='',
        tuning=tuning,
    )

    solver = DetSolver(cfg)
    solver.fit(progress_bar_epochs, progress_bar_iters, stop_button, charts_grid)

    return cfg


def setup_callbacks(
    iter_callback: Optional[Callable] = None, eval_callback: Optional[Callable] = None
):

    sly.logger.debug("Setting up callbacks...")

    def print_iter(logs: Logs):
        print("ITER | Iter IDX: ", logs.iter_idx)
        print("ITER | Loss, lrs, memory: ", logs.loss, logs.lrs, logs.cuda_memory)

    def print_eval(logs: Logs):
        # Charts: AP vs AR (maxDets=100), All APs, All ARs
        print("EVAL | Epoch: ", logs.epoch)
        print("EVAL | Metrics: ", logs.evaluation_metrics)

    if iter_callback is None:
        sly.logger.info("iter callback not provided, using default prints...")
        iter_callback = print_iter
    if eval_callback is None:
        sly.logger.info("eval callback not provided, using default prints...")
        eval_callback = print_eval

    LOGS.iter_callback = iter_callback
    LOGS.eval_callback = eval_callback
    sly.logger.debug("Callbacks set...")
    sly.logger.debug("Callbacks set...")
