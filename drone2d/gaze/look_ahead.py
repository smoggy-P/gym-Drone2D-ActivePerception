"""LookAhead gaze policy — point the camera along the velocity vector."""

from __future__ import annotations

import math

from drone2d.config import SimConfig


class LookAhead:
    """Orient the sensor in the direction of the drone's velocity.

    A simple but effective heuristic: the drone always looks where it is
    going.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.dt = config.dt

    def plan(self, state: dict) -> float:
        """Return a normalised yaw-rate command in ``[-1, 1]``.

        Parameters
        ----------
        state : dict
            Must contain ``'drone'`` — the :class:`~drone2d.drone.Drone2D`.
        """
        drone = state["drone"]
        vx, vy = drone.velocity
        if vx == 0 and vy == 0:
            return 0.0

        target_yaw = math.degrees(math.atan2(-vy, vx)) % 360
        diff = target_yaw - drone.yaw
        if abs(diff) < 180:
            yaw_vel = max(min(diff / self.dt, self.config.drone_max_yaw_speed),
                          -self.config.drone_max_yaw_speed)
        else:
            yaw_vel = -max(min(diff / self.dt, self.config.drone_max_yaw_speed),
                           -self.config.drone_max_yaw_speed)
        return yaw_vel / self.config.drone_max_yaw_speed
