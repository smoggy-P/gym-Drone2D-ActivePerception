"""Trajectory planner registry.

All concrete planners are registered in :data:`PLANNER_REGISTRY` and can be
looked up by name via :func:`get_planner`.

Example::

    planner_cls = get_planner("Primitive")
    planner = planner_cls(drone, config)
"""

from __future__ import annotations

from typing import Type

from drone2d.planners.base import Planner
from drone2d.planners.primitive import Primitive
from drone2d.planners.jerk_primitive import JerkPrimitive
from drone2d.planners.no_move import NoMove

# MPC requires the proprietary ``forcespro`` library — import lazily.
try:
    from drone2d.planners.mpc import MPC
except Exception:
    MPC = None  # type: ignore[assignment, misc]

PLANNER_REGISTRY: dict[str, Type[Planner]] = {
    "Primitive": Primitive,
    "Jerk_Primitive": JerkPrimitive,
    "NoMove": NoMove,
}
if MPC is not None:
    PLANNER_REGISTRY["MPC"] = MPC


def get_planner(name: str) -> Type[Planner]:
    """Return the planner class registered under *name*.

    Raises
    ------
    KeyError
        If no planner with that name exists.
    """
    if name not in PLANNER_REGISTRY:
        available = ", ".join(sorted(PLANNER_REGISTRY))
        raise KeyError(
            f"Unknown planner '{name}'. Available: {available}"
        )
    return PLANNER_REGISTRY[name]
