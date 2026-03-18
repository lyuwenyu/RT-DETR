# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .dtype import as_torch_dtype
from .utils import (
    cat_keep_shapes,
    count_parameters,
    fix_random_seeds,
    get_conda_env,
    get_sha,
    named_apply,
    named_replace,
    uncat_with_shapes,
)
