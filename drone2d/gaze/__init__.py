"""Gaze (yaw) planner registry.

All gaze control policies are registered in :data:`GAZE_REGISTRY` and can be
retrieved by name via :func:`get_gaze_planner`.

Example::

    gaze_cls = get_gaze_planner("Oxford")
    gaze = gaze_cls(config)
    action = gaze.plan(observation)
"""

from __future__ import annotations

from typing import Any, Type

from drone2d.gaze.look_ahead import LookAhead
from drone2d.gaze.no_control import NoControl
from drone2d.gaze.oxford import Oxford
from drone2d.gaze.rotating import Rotating
from drone2d.gaze.owl import Owl
from drone2d.gaze.look_goal import LookGoal

GAZE_REGISTRY: dict[str, Type[Any]] = {
    "LookAhead": LookAhead,
    "NoControl": NoControl,
    "Oxford": Oxford,
    "Rotating": Rotating,
    "Owl": Owl,
    "LookGoal": LookGoal,
}


def get_gaze_planner(name: str):
    """Return the gaze planner class registered under *name*.

    Raises
    ------
    KeyError
        If no gaze planner with that name exists.
    """
    if name not in GAZE_REGISTRY:
        available = ", ".join(sorted(GAZE_REGISTRY))
        raise KeyError(
            f"Unknown gaze method '{name}'. Available: {available}"
        )
    return GAZE_REGISTRY[name]
