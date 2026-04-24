from __future__ import annotations

import random

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)
