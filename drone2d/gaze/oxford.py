"""Oxford gaze policy — maximise observation of the swept trajectory.

Implements the information-gain-based gaze planning strategy inspired by
the Oxford Robotics Institute approach.  A reward map is computed from
recency of observation and proximity to the planned path, and the yaw
primitive that maximises the expected reward is selected.
"""

from __future__ import annotations

import math

import numpy as np

from drone2d.config import SimConfig
from drone2d.drone import Drone2D


class Oxford:
    """Information-gain gaze planner.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.dt = config.dt
        gw = config.map_size[0] // config.map_scale
        gh = config.map_size[1] // config.map_scale

        self._last_observed = 5.0 * np.ones((gw, gh))
        self._swept = np.zeros((gw, gh))
        self._dim = config.map_size

        # Planner hyper-parameters
        self._tau_s = 3       # priority horizon (steps)
        self._tau_c = 0.5     # safe observation recency
        self._c1 = 1_000_000
        self._c2 = 1_000
        self._c3 = 1

        self._v_yaw_space = np.arange(
            -config.drone_max_yaw_speed,
            config.drone_max_yaw_speed,
            config.drone_max_yaw_speed / 3,
        )

    # ------------------------------------------------------------------
    def plan(self, observation: dict) -> float:
        """Return normalised yaw-rate in ``[-1, 1]``."""
        drone = observation["drone"]
        trajectory = observation["trajectory"]

        ms = self.config.map_scale

        # Update swept map
        self._swept[:] = 0
        for i, pos in enumerate(trajectory.positions):
            self._swept[int(pos[0] // ms), int(pos[1] // ms)] = i * self.dt

        # Update last-observed map
        view = self._get_view_map(drone)
        self._last_observed = np.where(
            view, 0, self._last_observed + (1 - view) * self.dt,
        )

        # Reward map
        reward = np.where(
            (self._swept > 0) & (self._swept <= self._tau_s) & (self._last_observed >= self._tau_c),
            self._c1,
            np.where(
                (self._swept > self._tau_s) & (self._last_observed >= self._tau_c),
                self._c2,
                np.clip(self._c3 * self._last_observed, -np.inf, 1),
            ),
        )

        if len(trajectory) == 0:
            return 0.0

        # Evaluate yaw primitives
        target_yaws = drone.yaw + self._v_yaw_space * self.dt
        best_idx = 0
        best_reward = 0.0
        for i, yaw in enumerate(target_yaws):
            tmp_drone = Drone2D(
                trajectory.positions[0][0],
                trajectory.positions[0][1],
                yaw, self.dt, self.config,
            )
            vm = self._get_view_map(tmp_drone)
            r = np.sum(vm * reward)
            if r > best_reward:
                best_reward = r
                best_idx = i

        return self._v_yaw_space[best_idx] / self.config.drone_max_yaw_speed

    # ------------------------------------------------------------------
    def _get_view_map(self, drone) -> np.ndarray:
        """Binary map: 1 where the cell is within the drone's FOV."""
        ms = self.config.map_scale
        gw = self._dim[0] // ms
        gh = self._dim[1] // ms

        x = np.arange(gw).reshape(-1, 1) * ms
        y = np.arange(gh).reshape(1, -1) * ms

        vec_yaw = np.array([
            math.cos(math.radians(drone.yaw)),
            -math.sin(math.radians(drone.yaw)),
        ])
        half_fov = math.radians(drone.yaw_range / 2)

        dx = x - drone.x
        dy = y - drone.y
        dist_sq = dx ** 2 + dy ** 2

        np.seterr(divide="ignore", invalid="ignore")
        cos_angle = (dx * vec_yaw[0] + dy * vec_yaw[1]) / np.sqrt(dist_sq)
        in_fov = np.arccos(np.clip(cos_angle, -1, 1)) <= half_fov
        in_range = dist_sq <= drone.yaw_depth ** 2
        at_origin = dist_sq <= 0

        return np.where(np.logical_or(at_origin, np.logical_and(in_fov, in_range)), 1, 0)
