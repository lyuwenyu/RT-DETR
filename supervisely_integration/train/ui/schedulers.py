from typing import List

from supervisely.app.widgets import (
    Container,
    Empty,
    Field,
    Input,
    InputNumber,
    SelectString,
    Switch,
)
from supervisely_integration.train.ui.utils import (
    OrderedWidgetWrapper,
    create_linked_getter,
    get_switch_value,
    set_switch_value,
)

schedulers = [("empty", "Without scheduler")]


def get_multisteps(input_w: Input) -> List[int]:
    steps: str = input_w.get_value()
    return [int(st.strip()) for st in steps.split(",")]


def set_multisteps(input_w: Input, value: List[int]):
    input_w.set_value(",".join(value))


multi_steps_scheduler = OrderedWidgetWrapper("MultiStepLR")

multi_steps_input = Input("16,22")
multi_steps_field = Field(
    multi_steps_input,
    "LR sheduler steps",
    "Many int step values splitted by comma",
)
multi_steps_scheduler.add_input(
    "milestones",
    multi_steps_input,
    wraped_widget=multi_steps_field,
    custom_value_getter=get_multisteps,
    custom_value_setter=set_multisteps,
)

multi_steps_gamma_input = InputNumber(0.1, 0, step=1e-5, size="small")
multi_steps_gamma_field = Field(multi_steps_gamma_input, "Gamma")
multi_steps_scheduler.add_input("gamma", multi_steps_gamma_input, multi_steps_gamma_field)
schedulers.append((repr(multi_steps_scheduler), "Multistep LR"))

# CosineAnnealingLR
cosineannealing_scheduler = OrderedWidgetWrapper("CosineAnnealingLR")

cosineannealing_tmax_input = InputNumber(1, 1, step=1, size="small")
cosineannealing_tmax_field = Field(
    cosineannealing_tmax_input,
    "T max",
    "Maximum number of iterations",
)
cosineannealing_scheduler.add_input(
    "T_max",
    cosineannealing_tmax_input,
    cosineannealing_tmax_field,
)

etamin_switch_input = Switch(True)
etamin_input = InputNumber(0, 0, step=1e-6, size="small")
etamin_field = Field(
    Container([etamin_switch_input, etamin_input]),
    "Min LR",
    "Minimum learning rate",
)

etamin_ratio_input = InputNumber(0, 0, step=1e-6, size="small")
etamin_ratio_input.disable()
etamin_ratio_field = Field(
    etamin_ratio_input,
    "Min LR Ratio",
    "The ratio of the minimum parameter value to the base parameter value",
)

cosineannealing_scheduler.add_input(
    "eta_min",
    etamin_input,
    etamin_field,
    custom_value_getter=create_linked_getter(
        etamin_input,
        etamin_ratio_input,
        etamin_switch_input,
        True,
    ),
)

schedulers.append((repr(cosineannealing_scheduler), "Cosine Annealing LR"))


# LinearLR
linear_scheduler = OrderedWidgetWrapper("LinearLR")

linear_start_factor_input = InputNumber(0.333, 1e-4, step=1e-4)
linear_start_factor_field = Field(
    linear_start_factor_input,
    "Start factor",
    description=(
        "The number we multiply learning rate in the "
        "first epoch. The multiplication factor changes towards end factor "
        "in the following epochs"
    ),
)
linear_scheduler.add_input("start_factor", linear_start_factor_input, linear_start_factor_field)


linear_end_factor_input = InputNumber(1.0, 1e-4, step=1e-4)
linear_end_factor_field = Field(
    linear_end_factor_input,
    "End factor",
    description=("The number we multiply learning rate at the end " "of linear changing process"),
)
linear_scheduler.add_input("end_factor", linear_end_factor_input, linear_end_factor_field)
schedulers.append((repr(linear_scheduler), "Linear LR"))

# OneCycleLR
onecycle_scheduler = OrderedWidgetWrapper("OneCycleLR")

# TODO: теоретически этот параметр может быть списком. Надо ли?
onecycle_eta_input = InputNumber(1, 0, step=1e-6, size="small")
onecycle_eta_field = Field(
    onecycle_eta_input, "Max LR", "Upper parameter value boundaries in the cycle"
)
onecycle_scheduler.add_input(
    "max_lr",
    onecycle_eta_input,
    onecycle_eta_field,
)

onecycle_total_steps_input = InputNumber(100, 1, step=1, size="small")
onecycle_total_steps_field = Field(
    onecycle_total_steps_input, "Total steps", "The total number of steps in the cycle"
)
onecycle_scheduler.add_input(
    "total_steps",
    onecycle_total_steps_input,
    onecycle_total_steps_field,
)

onecycle_pct_start_input = InputNumber(0.3, 0, step=0.001)
onecycle_pct_start_field = Field(
    onecycle_pct_start_input,
    "Start percentage",
    "The percentage of the cycle (in number of steps) spent increasing the learning rate.",
)
onecycle_scheduler.add_input(
    "pct_start",
    onecycle_pct_start_input,
    onecycle_pct_start_field,
)

onecycle_anneal_strategy_input = SelectString(["cos", "linear"])
onecycle_anneal_strategy_field = Field(onecycle_anneal_strategy_input, "Anneal strategy")
onecycle_scheduler.add_input(
    "anneal_strategy",
    onecycle_anneal_strategy_input,
    onecycle_anneal_strategy_field,
    custom_value_getter=lambda w: w.get_value(),
    custom_value_setter=lambda w, v: w.set_value(v),
)

onecycle_div_factor_input = InputNumber(25, 1e-4, step=1e-3)
onecycle_div_factor_field = Field(
    onecycle_div_factor_input,
    "Div factor",
    "Determines the initial learning rate via initial_param = max_lr/div_factor",
)
onecycle_scheduler.add_input(
    "div_factor",
    onecycle_div_factor_input,
    onecycle_div_factor_field,
)

onecycle_findiv_factor_input = InputNumber(10000, 1e-6, step=1)
onecycle_findiv_factor_field = Field(
    onecycle_findiv_factor_input,
    "Final div factor",
    "Determines the minimum learning rate via min_lr = initial_param/final_div_factor",
)
onecycle_scheduler.add_input(
    "final_div_factor",
    onecycle_findiv_factor_input,
    onecycle_findiv_factor_field,
)


onecycle_three_phase_input = Switch(True)
onecycle_three_phase_field = Field(
    onecycle_three_phase_input,
    "Use three phase",
    (
        "If turned on, use a third phase of the schedule to "
        "annihilate the learning rate according to `final_div_factor` "
        "instead of modifying the second phase"
    ),
)
onecycle_scheduler.add_input(
    "three_phase",
    onecycle_three_phase_input,
    onecycle_three_phase_field,
    get_switch_value,
    set_switch_value,
)
schedulers.append((repr(onecycle_scheduler), "OneCycleLR"))

schedulers_params = {
    "Without scheduler": Empty(),
    repr(multi_steps_scheduler): multi_steps_scheduler,
    repr(cosineannealing_scheduler): cosineannealing_scheduler,
    repr(linear_scheduler): linear_scheduler,
    repr(onecycle_scheduler): onecycle_scheduler,
}
