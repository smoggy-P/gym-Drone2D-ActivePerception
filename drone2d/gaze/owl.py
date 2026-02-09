"""Owl gaze policy — multi-objective cost-based yaw planning.

Balances goal direction, velocity direction, obstacle tracking and
exploration when selecting a yaw action.
"""

from __future__ import annotations

from math import cos, sin, radians, degrees, atan2

import numpy as np
from numpy.linalg import norm

from drone2d.config import SimConfig


def _angle_between(a1: float, a2: float) -> float:
    """Shortest angular distance in degrees."""
    a1 %= 360
    a2 %= 360
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)


class Owl:
    """Multi-objective gaze planner with occupancy tracking.

    The cost function combines:
    - Goal direction tracking.
    - Velocity-aligned viewing.
    - Dynamic obstacle tracking.
    - Occupancy uncertainty.
    - Yaw-rate penalty.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self._dt_plan = 0.8  # planning horizon step (seconds)
        self._u_buffer: list[float] = []
        self._weights = np.array([0.2, 0.9, 1.0, 0.1, 0.0])

        self._u_space = np.arange(
            -config.drone_max_yaw_speed,
            config.drone_max_yaw_speed,
            config.drone_max_yaw_speed / 10,
        )
        self._theta_h = config.drone_view_range
        self._l_hit = 0.4
        self._l_miss = -0.05
        self._beta = 1.0
        self._U_list = np.zeros(36)

    # ------------------------------------------------------------------
    def plan(self, observation: dict) -> float:
        """Return normalised yaw-rate in ``[-1, 1]``."""
        # Consume buffered commands first
        if self._u_buffer:
            return self._u_buffer.pop() / self.config.drone_max_yaw_speed

        drone = observation["drone"]
        target = observation["target"]
        trackers = drone.trackers

        self._update_U(drone, self._dt_plan)

        d_g = degrees(atan2(target[1] - drone.y, target[0] - drone.x))
        d_v = degrees(atan2(*((drone.velocity / max(norm(drone.velocity), 1e-6))[::-1])))
        active_dirs = [
            degrees(atan2(*((t.mu_upds[-1][:2, 0] - np.array([drone.x, drone.y]))[::-1])))
            for t in trackers if t.active
        ]

        yaws = -(drone.yaw + self._u_space * self._dt_plan)
        f = np.zeros((len(yaws), 5))

        for i, yaw in enumerate(yaws):
            f[i, 0] = self._G(yaw - d_g) * (1 - self._U(d_g))
            f[i, 1] = (norm(drone.velocity / 10) ** 2) * self._G(yaw - d_v) * (1 - self._U(d_v))

            for d_o, tracker in zip(active_dirs, [t for t in trackers if t.active]):
                vel_mag = norm(tracker.mu_upds[-1][2:, 0])
                dist = norm(tracker.mu_upds[-1][:2, 0] - np.array([drone.x, drone.y]))
                f[i, 2] += self._beta * vel_mag / max(dist, 1e-6) * self._G(yaw - d_o)

            f[i, 3] = self._U(yaw)
            f[i, 4] = abs(radians(self._u_space[i] * self._dt_plan))

        costs = f @ self._weights
        idx = np.argmin(costs)

        # Buffer commands for sub-steps
        n_sub = int(self._dt_plan // self.config.dt) - 1
        self._u_buffer = [self._u_space[idx]] * n_sub

        return self._u_space[idx] / self.config.drone_max_yaw_speed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _G(self, theta: float) -> float:
        """Gaze cost: penalises directions outside the FOV."""
        if _angle_between(theta, 0) <= self._theta_h / 2:
            return 0.0
        return (
            radians(_angle_between(theta, self._theta_h / 2))
            * radians(_angle_between(theta, -self._theta_h / 2))
        )

    def _update_U(self, drone, dt: float) -> None:
        """Update directional occupancy uncertainty."""
        delta_p = drone.velocity * dt
        for i, d_i in enumerate(np.arange(0, 360, 10)):
            d_hat = np.array([cos(radians(d_i)), sin(radians(d_i))])
            L_yt = -delta_p.dot(d_hat) / self.config.drone_view_depth
            L_yt += self._l_hit if _angle_between(d_i, -drone.yaw) < self._theta_h / 2 else self._l_miss
            self._U_list[i] = np.clip(self._U_list[i] + L_yt, 0, 1)

    def _U(self, theta: float) -> float:
        """Lookup the occupancy uncertainty for direction *theta*."""
        idx = np.argmin([_angle_between(d, theta) for d in np.arange(0, 360, 10)])
        return self._U_list[idx]
