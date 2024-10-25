import numpy as np
import yaml

import supervisely as sly
import supervisely_integration.train.globals as g
import supervisely_integration.train.ui.output as output_ui
import supervisely_integration.train.ui.schedulers as schedulers
import supervisely_integration.train.ui.utils as utils_ui
import supervisely_integration.train.utils as utils
from supervisely.app.widgets import (
    BindedInputNumber,
    Button,
    Card,
    Checkbox,
    Container,
    Editor,
    Empty,
    Field,
    Input,
    InputNumber,
    LineChart,
    Select,
    Switch,
    Tabs,
    Text,
)

# region advanced widgets
advanced_mode_checkbox = Checkbox("Advanced mode")
advanced_mode_field = Field(
    advanced_mode_checkbox,
    title="Advanced mode",
    description="Enable advanced mode to specify custom training parameters manually.",
)
with open(g.default_config_path, "r") as f:
    default_config = f.read()
    sly.logger.debug(f"Loaded default config from {g.default_config_path}")
advanced_mode_editor = Editor(default_config, language_mode="yaml", height_lines=100)
advanced_mode_editor.hide()
# endregion

# region general widgets
number_of_epochs_input = InputNumber(value=20, min=1)
number_of_epochs_field = Field(
    number_of_epochs_input,
    title="Number of epochs",
    description="The number of epochs to train the model for",
)
input_size_input = BindedInputNumber(640, 640)
input_size_field = Field(
    input_size_input,
    title="Input size",
    description="Images will be resized to this size.",
)

train_batch_size_input = InputNumber(value=4, min=1)
train_batch_size_field = Field(
    train_batch_size_input,
    title="Train batch size",
    description="The number of images in a batch during training",
)

val_batch_size_input = InputNumber(value=8, min=1)
val_batch_size_field = Field(
    val_batch_size_input,
    title="Validation batch size",
    description="The number of images in a batch during validation",
)

validation_interval_input = InputNumber(value=1, min=1)
validation_interval_field = Field(
    validation_interval_input,
    title="Validation interval",
    description="The number of epochs between each validation run",
)

checkpoints_interval_input = InputNumber(value=1, min=1)
checkpoints_interval_field = Field(
    checkpoints_interval_input,
    title="Checkpoint interval",
    description="The number of epochs between each checkpoint save",
)

general_tab = Container(
    [
        number_of_epochs_field,
        input_size_field,
        train_batch_size_field,
        val_batch_size_field,
        validation_interval_field,
        checkpoints_interval_field,
    ]
)
# endregion

# region optimizer widgets
optimizer_select = Select([Select.Item(opt) for opt in g.OPTIMIZERS])
optimizer_field = Field(
    optimizer_select,
    title="Select optimizer",
    description="Choose the optimizer to use for training",
)
learning_rate_input = InputNumber(value=0.0002, step=0.00005)
learning_rate_field = Field(
    learning_rate_input,
    title="Learning rate",
    description="The learning rate to use for the optimizer",
)
wight_decay_input = InputNumber(value=0.0001, step=0.00001)
wight_decay_field = Field(
    wight_decay_input,
    title="Weight decay",
    description="The amount of L2 regularization to apply to the weights",
)
momentum_input = InputNumber(value=0.9, step=0.1)
momentum_field = Field(
    momentum_input,
    title="Momentum",
    description="The amount of momentum to apply to the weights",
)
momentum_field.hide()
beta1_input = InputNumber(value=0.9, step=0.1)
beta1_field = Field(
    beta1_input,
    title="Beta 1",
    description="The exponential decay rate for the first moment estimates",
)
beta2_input = InputNumber(value=0.999, step=0.001)
beta2_field = Field(
    beta2_input,
    title="Beta 2",
    description="The exponential decay rate for the second moment estimates",
)

clip_gradient_norm_checkbox = Checkbox("Clip gradient norm")
clip_gradient_norm_input = InputNumber(value=0.1, step=0.01)
clip_gradient_norm_field = Field(
    Container([clip_gradient_norm_checkbox, clip_gradient_norm_input]),
    title="Clip gradient norm",
    description="Select the highest gradient norm to clip the gradients",
)


optimization_tab = Container(
    [
        optimizer_field,
        learning_rate_field,
        wight_decay_field,
        momentum_field,
        beta1_field,
        beta2_field,
        clip_gradient_norm_field,
    ]
)

# endregion

# region scheduler widgets
scheduler_preview_chart = LineChart(
    "Scheduler preview", stroke_curve="straight", height=400, decimalsInFloat=6, markers_size=0
)
scheduler_preview_chart.hide()
scheduler_preview_btn = Button("Preview", button_size="small")
scheduler_clear_btn = Button("Clear", button_size="small", plain=True)
scheduler_preview_info = Text("", status="info")


select_scheduler_items = [Select.Item(val, label) for val, label in schedulers.schedulers]
select_scheduler = Select(items=select_scheduler_items)
select_scheduler_field = Field(
    select_scheduler,
    title="Select scheduler",
)

enable_warmup_input = Switch(True)
enable_warmup_field = Field(enable_warmup_input, "Enable warmup")

warmup = utils_ui.OrderedWidgetWrapper("warmup")
warmup_iterations_input = InputNumber(25, 1, step=1)
warmup_iterations_field = Field(
    warmup_iterations_input, "Warmup iterations", "The number of iterations that warmup lasts"
)
warmup.add_input("warmup_iters", warmup_iterations_input, warmup_iterations_field)

warmup_ratio = InputNumber(0.001, step=0.0001)
warmup_ratio_field = Field(
    warmup_ratio,
    "Warmup ratio",
    "LR used at the beginning of warmup equals to warmup_ratio * initial_lr",
)
warmup.add_input("warmup_ratio", warmup_ratio, warmup_ratio_field)

learning_rate_scheduler_tab = Container(
    [
        select_scheduler_field,
        Container(
            [
                schedulers.multi_steps_scheduler.create_container(hide=True),
                schedulers.cosineannealing_scheduler.create_container(hide=True),
                schedulers.linear_scheduler.create_container(hide=True),
                schedulers.onecycle_scheduler.create_container(hide=True),
            ],
            gap=0,
        ),
        enable_warmup_field,
        warmup.create_container(),
        scheduler_preview_chart,
        Container(
            [scheduler_preview_btn, scheduler_clear_btn, Empty()],
            "horizontal",
            0,
            fractions=[1, 1, 10],
        ),
    ],
)

# endregion

select_params_button = Button("Select")

parameters_tabs = Tabs(
    ["General", "Optimizer (Advanced)", "Learning rate scheduler (Advanced)"],
    contents=[
        general_tab,
        # checkpoints_tab,
        optimization_tab,
        learning_rate_scheduler_tab,
    ],
)

# Model Benchmark
run_model_benchmark_checkbox = Checkbox(content="Run Model Benchmark evaluation", checked=True)
run_speedtest_checkbox = Checkbox(content="Run speed test", checked=True)

model_benchmark_f = Field(
    Container(
        widgets=[
            run_model_benchmark_checkbox,
            run_speedtest_checkbox,
        ]
    ),
    title="Model Evaluation Benchmark",
    description=f"Generate evalutaion dashboard with visualizations and detailed analysis of the model performance after training. The best checkpoint will be used for evaluation. You can also run speed test to evaluate model inference speed.",
)
docs_link = '<a href="https://docs.supervisely.com/neural-networks/model-evaluation-benchmark/" target="_blank">documentation</a>'
model_benchmark_learn_more = Text(
    f"Learn more about Model Benchmark in the {docs_link}.", status="info"
)


content = Container(
    [
        advanced_mode_field,
        advanced_mode_editor,
        parameters_tabs,
        model_benchmark_f,
        model_benchmark_learn_more,
        select_params_button,
    ],
)

card = Card(
    title="Training hyperparameters",
    description="Specify training hyperparameters using one of the methods.",
    content=content,
    lock_message="Select augmentations to unlock",
)
card.lock()


@run_model_benchmark_checkbox.value_changed
def change_model_benchmark(value):
    if value:
        run_speedtest_checkbox.show()
    else:
        run_speedtest_checkbox.hide()


@advanced_mode_checkbox.value_changed
def advanced_mode_changed(is_checked: bool):
    if is_checked:
        advanced_mode_editor.show()
        parameters_tabs.hide()
    else:
        advanced_mode_editor.hide()
        parameters_tabs.show()


@optimizer_select.value_changed
def optimizer_changed(optimizer: str):
    if optimizer == "Adam":
        beta1_field.show()
        beta2_field.show()
        momentum_field.hide()
    elif optimizer == "AdamW":
        beta1_field.hide()
        beta2_field.hide()
        momentum_field.hide()
    elif optimizer == "SGD":
        beta1_field.hide()
        beta2_field.hide()
        momentum_field.show()


@enable_warmup_input.value_changed
def warmup_changed(is_switched: bool):
    if is_switched:
        warmup_iterations_field.show()
        warmup_iterations_input.show()
        warmup_ratio_field.show()
        warmup_ratio.show()
    else:
        warmup_iterations_field.hide()
        warmup_iterations_input.hide()
        warmup_ratio_field.hide()
        warmup_ratio.hide()


@scheduler_preview_btn.click
def on_preview_scheduler():
    import torch
    import visualize_scheduler
    from torch.optim import SGD

    from rtdetr_pytorch.utils import name2cls

    total_epochs = number_of_epochs_input.get_value()
    batch_size = train_batch_size_input.get_value()
    start_lr = learning_rate_input.get_value()
    total_images = g.project_info.items_count

    from supervisely.app.widgets import TrainValSplits
    from supervisely_integration.train.ui.splits import trainval_splits

    split_widget: TrainValSplits = trainval_splits
    # split_widget.get_splits()
    self = split_widget
    split_method = self._content.get_active_tab()
    if split_method == "Random":
        splits_counts = self._random_splits_table.get_splits_counts()
        train_count = splits_counts["train"]
        val_count = splits_counts["val"]
        val_part = val_count / (val_count + train_count)
        n_images = total_images
        val_count = round(val_part * n_images)
        train_count = n_images - val_count
    elif split_method == "Based on item tags":
        # can't predict
        train_count = total_images
    elif split_method == "Based on datasets":
        train_ds_ids = self._train_ds_select.get_selected_ids()
        val_ds_ids = self._val_ds_select.get_selected_ids()
        ds_infos = self._api.dataset.get_list(self._project_id)
        train_ds_names, val_ds_names = [], []
        train_count, val_count = 0, 0
        for ds_info in ds_infos:
            if ds_info.id in train_ds_ids:
                train_ds_names.append(ds_info.name)
                train_count += ds_info.items_count
            if ds_info.id in val_ds_ids:
                val_ds_names.append(ds_info.name)
                val_count += ds_info.items_count

    dataloader_len = train_count // batch_size

    def instantiate(d, **cls_kwargs):
        d = d.copy()
        cls = name2cls(d.pop("type"))
        return cls(**{**d, **cls_kwargs})

    custom_config = read_parameters(train_count)
    lr_scheduler = custom_config.get("lr_scheduler")
    dummy_optim = SGD([torch.nn.Parameter(torch.tensor([5.0]))], start_lr)
    if lr_scheduler is not None:
        lr_scheduler = instantiate(lr_scheduler, optimizer=dummy_optim)

    use_lr_warmup = enable_warmup_input.is_switched()
    if use_lr_warmup:
        lr_warmup = custom_config.get("lr_warmup")
        if lr_warmup is not None:
            lr_warmup = instantiate(lr_warmup, optimizer=dummy_optim)
    else:
        lr_warmup = None

    x, lrs = visualize_scheduler.test_schedulers(
        lr_scheduler, lr_warmup, dummy_optim, dataloader_len, total_epochs
    )

    scheduler_name = select_scheduler.get_value()
    if scheduler_name == "empty":
        scheduler_name = "Without scheduler"

    name = f"{scheduler_name} warmup" if use_lr_warmup else scheduler_name
    scheduler_preview_chart.add_series(f"{name}", x, lrs)
    scheduler_preview_chart.show()
    scheduler_preview_info.set(
        f"Estimated train images count = {train_count}. Actual curve can be different.",
        status="info",
    )


@scheduler_clear_btn.click
def on_clear_preview():
    scheduler_preview_chart.set_series([])


@select_scheduler.value_changed
def update_scheduler(new_value):
    for scheduler in schedulers.schedulers_params.keys():
        if new_value == scheduler:
            schedulers.schedulers_params[scheduler].show()
        else:
            schedulers.schedulers_params[scheduler].hide()


@select_params_button.click
def select_params():
    if select_params_button.text == "Select":
        # widgets to disable
        utils_ui.disable_enable(
            [
                advanced_mode_checkbox,
                advanced_mode_editor,
                number_of_epochs_input,
                input_size_input,
                train_batch_size_input,
                val_batch_size_input,
                validation_interval_input,
                checkpoints_interval_input,
                optimizer_select,
                learning_rate_input,
                wight_decay_input,
                momentum_input,
                beta1_input,
                beta2_input,
                clip_gradient_norm_checkbox,
                clip_gradient_norm_input,
                warmup_iterations_input,
                scheduler_preview_chart,
                scheduler_preview_btn,
                scheduler_clear_btn,
                select_scheduler,
                enable_warmup_input,
                warmup,
                warmup_iterations_input,
                warmup_ratio,
                parameters_tabs,
                run_model_benchmark_checkbox,
                run_speedtest_checkbox,
            ],
            True,
        )

        utils_ui.update_custom_button_params(select_params_button, utils_ui.reselect_params)
        g.update_step()

        # unlock
        output_ui.card.unlock()
    else:
        # lock
        output_ui.card.lock()
        utils_ui.update_custom_button_params(select_params_button, utils_ui.select_params)

        # widgets to enable
        utils_ui.disable_enable(
            [
                advanced_mode_checkbox,
                advanced_mode_editor,
                number_of_epochs_input,
                input_size_input,
                train_batch_size_input,
                val_batch_size_input,
                validation_interval_input,
                checkpoints_interval_input,
                optimizer_select,
                learning_rate_input,
                wight_decay_input,
                momentum_input,
                beta1_input,
                beta2_input,
                clip_gradient_norm_checkbox,
                clip_gradient_norm_input,
                warmup_iterations_input,
                scheduler_preview_chart,
                scheduler_preview_btn,
                scheduler_clear_btn,
                select_scheduler,
                enable_warmup_input,
                warmup,
                warmup_iterations_input,
                warmup_ratio,
                parameters_tabs,
                run_model_benchmark_checkbox,
                run_speedtest_checkbox,
            ],
            False,
        )
        g.update_step(back=True)


def read_parameters(train_items_cnt: int):
    sly.logger.debug("Reading training parameters...")
    if advanced_mode_checkbox.is_checked():
        sly.logger.info("Advanced mode enabled, using custom config from the editor.")
        custom_config = advanced_mode_editor.get_value()
    else:
        sly.logger.info("Advanced mode disabled, reading parameters from the widgets.")
        with open(g.default_config_path, "r") as f:
            custom_config = f.read()
        custom_config = yaml.safe_load(custom_config)

        clip_max_norm = (
            clip_gradient_norm_input.get_value() if clip_gradient_norm_checkbox.is_checked() else -1
        )
        general_params = {
            "epoches": number_of_epochs_input.value,
            "val_step": validation_interval_input.value,
            "checkpoint_step": checkpoints_interval_input.value,
            "clip_max_norm": clip_max_norm,
        }

        total_steps = general_params["epoches"] * np.ceil(
            train_items_cnt / train_batch_size_input.value
        )

        optimizer_params = read_optimizer_parameters()
        scheduler_params, scheduler_cls_params = read_scheduler_parameters(total_steps)

        sly.logger.debug(f"General parameters: {general_params}")
        sly.logger.debug(f"Optimizer parameters: {optimizer_params}")
        sly.logger.debug(f"Scheduler parameters: {scheduler_cls_params}")

        custom_config.update(general_params)
        custom_config["optimizer"]["type"] = optimizer_params["optimizer"]
        custom_config["optimizer"]["lr"] = optimizer_params["learning_rate"]
        custom_config["optimizer"]["weight_decay"] = optimizer_params["weight_decay"]
        if optimizer_params.get("momentum"):
            custom_config["optimizer"]["momentum"] = optimizer_params["momentum"]
        else:
            custom_config["optimizer"]["betas"] = [
                optimizer_params["beta1"],
                optimizer_params["beta2"],
            ]

        # Set input_size
        w, h = input_size_input.get_value()
        for op in custom_config["train_dataloader"]["dataset"]["transforms"]["ops"]:
            if op["type"] == "Resize":
                op["size"] = [w, h]
        for op in custom_config["val_dataloader"]["dataset"]["transforms"]["ops"]:
            if op["type"] == "Resize":
                op["size"] = [w, h]
        if "HybridEncoder" in custom_config:
            custom_config["HybridEncoder"]["eval_spatial_size"] = [w, h]
        else:
            custom_config["HybridEncoder"] = {"eval_spatial_size": [w, h]}
        if "RTDETRTransformer" in custom_config:
            custom_config["RTDETRTransformer"]["eval_spatial_size"] = [w, h]
        else:
            custom_config["RTDETRTransformer"] = {"eval_spatial_size": [w, h]}

        custom_config["train_dataloader"]["batch_size"] = train_batch_size_input.value
        custom_config["val_dataloader"]["batch_size"] = val_batch_size_input.value
        custom_config["train_dataloader"]["num_workers"] = utils.get_num_workers(
            train_batch_size_input.value
        )
        custom_config["val_dataloader"]["num_workers"] = utils.get_num_workers(
            val_batch_size_input.value
        )

        # LR scheduler
        if scheduler_params["type"] == "Without scheduler":
            custom_config["lr_scheduler"] = None
        else:
            custom_config["lr_scheduler"] = scheduler_cls_params

        if scheduler_params["enable_warmup"]:
            custom_config["lr_warmup"] = {
                "type": "LinearLR",
                "total_iters": scheduler_params["warmup_iterations"],
                "start_factor": 0.001,
                "end_factor": 1.0,
            }

    return custom_config


def read_optimizer_parameters():
    optimizer = optimizer_select.get_value()

    parameters = {
        "optimizer": optimizer,
        "learning_rate": learning_rate_input.get_value(),
        "weight_decay": wight_decay_input.get_value(),
        "clip_gradient_norm": clip_gradient_norm_checkbox.is_checked(),
        "clip_gradient_norm_value": clip_gradient_norm_input.get_value(),
    }

    if optimizer in ["Adam", "AdamW"]:
        parameters.update(
            {
                "beta1": beta1_input.get_value(),
                "beta2": beta2_input.get_value(),
            }
        )
    elif optimizer == "SGD":
        parameters.update({"momentum": momentum_input.get_value()})

    return parameters


def read_scheduler_parameters(total_steps: int):
    scheduler = select_scheduler.get_value()
    if scheduler == "empty":
        scheduler = "Without scheduler"

    parameters = {
        "type": scheduler,
        "enable_warmup": enable_warmup_input.is_switched(),
        "warmup_iterations": warmup_iterations_input.get_value(),
    }

    if scheduler == "Without scheduler":
        return parameters, {}

    scheduler_cls_params = {
        "type": scheduler,
    }

    scheduler_widgets = schedulers.schedulers_params[scheduler]._widgets
    if scheduler_widgets is not None:
        for key, widget in scheduler_widgets.items():
            if isinstance(widget, (InputNumber, Input, Select)):
                scheduler_cls_params[key] = widget.get_value()
            elif isinstance(widget, Checkbox):
                scheduler_cls_params[key] = widget.is_checked()
            elif isinstance(widget, Switch):
                if not key == "by_epoch":
                    scheduler_cls_params[key] = widget.is_switched()

    if scheduler_cls_params["type"] == "OneCycleLR":
        scheduler_cls_params["total_steps"] = int(total_steps)
    elif scheduler_cls_params["type"] == "LinearLR":
        scheduler_cls_params["total_iters"] = int(total_steps)
    elif scheduler_cls_params["type"] == "MultiStepLR":
        scheduler_cls_params["milestones"] = [
            int(step.strip()) for step in scheduler_cls_params["milestones"].split(",")
        ]

    return parameters, scheduler_cls_params
