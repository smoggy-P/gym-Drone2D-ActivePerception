"""No-movement planner — keeps the drone stationary.

Used as a baseline when only the gaze policy is being evaluated.
"""

from __future__ import annotations

import numpy as np

from drone2d.config import SimConfig
from drone2d.drone import Drone2D
from drone2d.planners.base import Planner


class NoMove(Planner):
    """Planner that never moves the drone."""

    def plan(self, drone: Drone2D, update_dt: float) -> bool:
        self.target = np.array([-1.0, -1.0, 0.0, 0.0])
        return True

    def replan_check(self, drone: Drone2D) -> tuple[bool, np.ndarray]:
        return False, drone.map.grid_map
