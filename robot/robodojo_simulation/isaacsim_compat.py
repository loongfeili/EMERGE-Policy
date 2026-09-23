"""Work around Isaac Sim reading articulation properties off the GPU.

``SingleArticulation.dof_properties`` builds a NumPy structured array out of
values fetched from the articulation view, but the view's getters are not
consistent about which backend they answer with: on a CUDA device
``get_max_efforts()`` returns a torch tensor while the getters just above it
return arrays. Assigning that tensor into the array triggers an implicit
``.numpy()`` and raises

    TypeError: can't convert cuda:3 device type tensor to numpy

Nothing of ours is on the stack -- this fires inside ``env.reset()`` before the
driver is even constructed -- so it has to be corrected at the source. It cost
7 of 54 RoboDojo tasks a whole episode each, every task holding an articulated
object (``swap_blocks``, ``make_toast``, ``fill_egg_holder``, ...).
"""

from __future__ import annotations

import numpy as np

from robot.robodojo_simulation.tensors import to_numpy

_DOF_DTYPE = np.dtype(
    [
        ("type", int),
        ("hasLimits", bool),
        ("lower", float),
        ("upper", float),
        ("driveMode", int),
        ("maxVelocity", float),
        ("maxEffort", float),
        ("stiffness", float),
        ("damping", float),
    ]
)

_applied = False


def _dof_properties(self) -> np.ndarray:
    """``SingleArticulation.dof_properties`` with every read moved to the host."""
    view = self._articulation_view
    properties = np.zeros(self.num_dof, dtype=_DOF_DTYPE)
    limits = to_numpy(view.get_dof_limits()[0])
    properties["type"] = to_numpy(view.get_dof_types()[0])
    properties["lower"] = limits[:, 0]
    properties["upper"] = limits[:, 1]
    properties["hasLimits"] = properties["lower"] < properties["upper"]
    properties["driveMode"] = to_numpy(view.get_drive_types()[0])
    properties["maxEffort"] = to_numpy(view.get_max_efforts()[0])
    properties["maxVelocity"] = to_numpy(view.get_joint_max_velocities()[0])
    stiffnesses, dampings = view.get_gains()
    properties["stiffness"] = to_numpy(stiffnesses[0])
    properties["damping"] = to_numpy(dampings[0])
    return properties


def apply_isaacsim_compat() -> bool:
    """Install the patch. Returns whether it is now in place.

    Safe to call more than once, and a no-op when Isaac Sim is absent so the
    non-simulator tooling can import this module outside the simulator venv.
    """
    global _applied
    if _applied:
        return True
    try:
        from isaacsim.core.prims import SingleArticulation
    except Exception:
        return False
    SingleArticulation.dof_properties = property(_dof_properties)
    _applied = True
    return True


def select_planner_device(device_id: int) -> None:
    """Keep CuRobo on this worker's L20 when Vulkan requires visible GPUs."""
    from functools import partial
    import torch
    import warp as wp
    from curobo.types import DeviceCfg
    from env.planner_manager import curobo_planner
    device = torch.device("cuda", device_id)
    torch.cuda.set_device(device)
    wp.set_device(str(device))
    curobo_planner.DeviceCfg = partial(DeviceCfg, device=device)
