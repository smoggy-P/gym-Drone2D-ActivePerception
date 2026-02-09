"""Base class for all trajectory planners.

Every concrete planner (Primitive, MPC, JerkPrimitive, …) should inherit
from :class:`Planner` and implement :meth:`plan` and :meth:`replan_check`.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from drone2d.config import SimConfig
from drone2d.drone import Drone2D
from drone2d.trajectory import Trajectory2D


class Planner:
    """Abstract base for trajectory planners.

    Parameters
    ----------
    drone : Drone2D
        Reference drone (used only for initial state).
    config : SimConfig
        Simulation parameters.
    """

    def __init__(self, drone: Drone2D, config: SimConfig) -> None:
        self.trajectory = Trajectory2D()
        self.config = config
        self.target = np.array([drone.x, drone.y, 0.0, 0.0])

    def set_target(self, target) -> None:
        """Set the next navigation goal ``[x, y]``."""
        self.target = np.zeros(4)
        self.target[:2] = np.asarray(target)

    def is_free(
        self,
        position: np.ndarray,
        t: float,
        occupancy_map,
        trackers: list,
    ) -> bool:
        """Check whether *position* is collision-free at time *t*.

        The check considers static grid obstacles and predicted dynamic
        obstacle positions from the Kalman filter *trackers*.
        """
        if np.isnan(position).any():
            return False

        safe = self.config.drone_radius + 10
        offsets = [(-safe, 0), (0, 0), (safe, 0), (0, -safe), (0, safe)]
        for ox, oy in offsets:
            if occupancy_map.get_grid(position[0] + ox, position[1] + oy) == 1:
                return False

        for tracker in trackers:
            if tracker.active:
                pred_pos = tracker.estimate_pos(t)
                dist = norm(position - pred_pos)
                min_sep = self.config.drone_radius + tracker.radius + 5 + self.config.var_cam
                if dist <= min_sep:
                    return False
        return True

    def plan(self, drone: Drone2D, update_dt: float) -> bool:
        """Generate or extend the trajectory.

        Returns *True* on success, *False* if planning failed.
        """
        raise NotImplementedError

    def replan_check(self, drone: Drone2D) -> tuple[bool, np.ndarray]:
        """Check whether a replan is needed.

        Returns ``(needs_replan, swept_map)``.
        """
        raise NotImplementedError
