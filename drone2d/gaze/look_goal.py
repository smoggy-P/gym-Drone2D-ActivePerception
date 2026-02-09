"""LookGoal gaze policy — look towards unexplored cells along the path."""

from __future__ import annotations

import math

from drone2d.config import GRID_UNEXPLORED, SimConfig


class LookGoal:
    """Point the sensor towards the first unexplored cell on the trajectory.

    If all cells along the trajectory are explored, look at the trajectory
    endpoint (goal).
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config

    def plan(self, observation: dict) -> float:
        """Return normalised yaw-rate in ``[-1, 1]``."""
        trajectory = observation["trajectory"]
        drone = observation["drone"]

        if len(trajectory) == 0:
            return 0.0

        # Default: look at trajectory end
        x_look = trajectory.positions[-1][0]
        y_look = trajectory.positions[-1][1]

        # Override with first unexplored cell along trajectory
        for pos in trajectory.positions:
            if drone.map.get_grid(pos[0], pos[1]) == GRID_UNEXPLORED:
                x_look, y_look = pos[0], pos[1]
                break

        target_yaw = math.degrees(
            math.atan2(-(y_look - drone.y), x_look - drone.x)
        ) % 360

        diff = target_yaw - drone.yaw
        if abs(diff) < 180:
            yaw_vel = max(
                min(diff / self.config.dt, self.config.drone_max_yaw_speed),
                -self.config.drone_max_yaw_speed,
            )
        else:
            yaw_vel = -max(
                min(diff / self.config.dt, self.config.drone_max_yaw_speed),
                -self.config.drone_max_yaw_speed,
            )
        return yaw_vel / self.config.drone_max_yaw_speed
