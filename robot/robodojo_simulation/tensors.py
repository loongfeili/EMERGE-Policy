"""Convert whatever RoboDojo hands back into a NumPy array.

Some tasks return poses and joint states as CUDA tensors rather than arrays --
``fill_egg_holder`` and ``make_toast`` do, ``stack_bowls`` does not -- and a
bare ``np.asarray`` on one of those raises

    TypeError: can't convert cuda:6 device type tensor to numpy

part-way through an episode. Every value read out of the simulator goes through
here so the difference stops being visible above this line.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def to_numpy(value: Any) -> np.ndarray:
    """Detach, move to host, and convert. Plain arrays pass straight through."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def to_float_array(value: Any, dtype: Any = np.float64) -> np.ndarray:
    """``to_numpy`` plus a dtype, for the numeric reads that dominate callers."""
    return to_numpy(value).astype(dtype, copy=False)
