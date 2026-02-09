"""Drone2D — the 2-D quadrotor model.

:class:`Drone2D` represents a top-down quadrotor with limited field-of-view
sensor, occupancy grid, and per-agent Kalman filter trackers.
"""

from __future__ import annotations

import math
from math import cos, sin

import numpy as np
import pygame
from numpy.linalg import norm

from drone2d.config import GRID_OCCUPIED, SimConfig
from drone2d.grid_map import OccupancyGridMap
from drone2d.kalman_filter import KalmanFilter
from drone2d.raycast import Raycast
from drone2d.trajectory import Trajectory2D


MAX_TRACKERS = 500


class Drone2D:
    """Top-down 2-D quadrotor with perception and tracking.

    Parameters
    ----------
    init_x, init_y : float
        Starting position in world coordinates.
    init_yaw : float
        Starting yaw angle in degrees.
    dt : float
        Simulation time-step.
    config : SimConfig
        Full simulation configuration.
    """

    def __init__(
        self,
        init_x: float,
        init_y: float,
        init_yaw: float,
        dt: float,
        config: SimConfig,
    ) -> None:
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw % 360
        self.yaw_range = config.drone_view_range
        self.yaw_depth = config.drone_view_depth
        self.radius = config.drone_radius
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.dt = dt
        self.config = config

        # Perceived occupancy grid (starts unexplored)
        self.map = OccupancyGridMap(config.map_scale, config.map_size, 0)

        # Ray-casting sensor
        self.rays: list[dict] = []
        self.raycast = Raycast(
            plane_size=tuple(config.map_size),
            fov_deg=config.drone_view_range,
            depth=config.drone_view_depth,
            noise_sigma=config.var_cam,
        )

        # Kalman filter trackers (one per potential agent)
        self.trackers: list[KalmanFilter] = [
            KalmanFilter(config) for _ in range(MAX_TRACKERS)
        ]

    # ------------------------------------------------------------------
    # Motion
    # ------------------------------------------------------------------
    def step_pos(self, trajectory: Trajectory2D) -> None:
        """Advance position by consuming the next waypoint from *trajectory*."""
        if len(trajectory) > 0:
            self.acceleration = trajectory.accelerations[0]
            self.velocity = trajectory.velocities[0]
            self.x = round(trajectory.positions[0][0])
            self.y = round(trajectory.positions[0][1])
            trajectory.pop_front()

    def step_yaw(self, yaw_speed: float) -> None:
        """Rotate yaw by *yaw_speed* (degrees / s) for one time-step."""
        self.yaw = (self.yaw + yaw_speed * self.dt) % 360

    def brake(self) -> None:
        """Apply maximum deceleration to bring the drone to a stop."""
        max_decel = self.config.drone_max_acceleration * self.dt
        if norm(self.velocity) <= max_decel:
            self.velocity = np.zeros(2)
        else:
            direction = self.velocity / norm(self.velocity)
            self.velocity -= direction * max_decel
            self.x += self.velocity[0] * self.dt
            self.y += self.velocity[1] * self.dt

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def get_measurements(self, gt_map, agents: list):
        """Cast rays and return per-agent measurements.

        Returns
        -------
        newly_tracked : int
            Number of agents detected for the first time this step.
        measurements : list[np.ndarray | None]
            Noisy position measurement per agent, or *None*.
        """
        self.rays, newly_tracked, measurements = self.raycast.cast_rays(
            self, gt_map, agents,
        )
        return newly_tracked, measurements

    def update_trackers(self, measurements: list) -> list[KalmanFilter]:
        """Run one Kalman filter update for each agent.

        Returns
        -------
        list[KalmanFilter]
            Completed (archived) tracker instances.
        """
        achieved: list[KalmanFilter] = []
        for i, meas in enumerate(measurements):
            achieved.extend(self.trackers[i].update(meas))
        return achieved

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------
    def is_colliding(self, gt_map, agents: list) -> int:
        """Check for collisions with static obstacles or agents.

        Returns
        -------
        int
            ``0`` — no collision, ``1`` — static collision, ``2`` — dynamic
            collision.
        """
        pos = np.array([self.x, self.y])
        offsets = [
            (-self.radius, 0), (0, 0), (self.radius, 0),
            (0, -self.radius), (0, self.radius),
        ]
        for ox, oy in offsets:
            if gt_map.get_grid(pos[0] + ox, pos[1] + oy) == GRID_OCCUPIED:
                return 1

        for agent in agents:
            if norm(agent.position - pos) < agent.radius + self.radius:
                return 2

        return 0

    # ------------------------------------------------------------------
    # Local map extraction (for RL observations)
    # ------------------------------------------------------------------
    def get_local_map(self) -> np.ndarray:
        """Extract a square local map centred on the drone."""
        di = int(self.x // self.config.map_scale)
        dj = int(self.y // self.config.map_scale)
        half = 2 * (self.config.drone_view_depth // self.config.map_scale)
        padded = np.pad(
            self.map.grid_map,
            ((half, half), (half, half)),
            mode="constant",
            constant_values=0,
        )
        return padded[di: di + 2 * half + 1, dj: dj + 2 * half + 1]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, surface: pygame.Surface) -> None:
        """Draw the drone, its body and sensor cone onto *surface*."""
        color = (100, 100, 100)

        # Sensor cone arc
        pygame.draw.arc(
            surface,
            color,
            [
                self.x - self.yaw_depth,
                self.y - self.yaw_depth,
                2 * self.yaw_depth,
                2 * self.yaw_depth,
            ],
            math.radians(self.yaw - self.yaw_range / 2),
            math.radians(self.yaw + self.yaw_range / 2),
            2,
        )
        a1 = math.radians(self.yaw + self.yaw_range / 2)
        a2 = math.radians(self.yaw - self.yaw_range / 2)
        pygame.draw.line(
            surface, color,
            (self.x, self.y),
            (self.x + self.yaw_depth * cos(a1), self.y - self.yaw_depth * sin(a1)),
            2,
        )
        pygame.draw.line(
            surface, color,
            (self.x, self.y),
            (self.x + self.yaw_depth * cos(a2), self.y - self.yaw_depth * sin(a2)),
            2,
        )

        # Drone body (cross + rotors)
        dx = 5 * cos(math.radians(self.yaw + 45))
        dy = -5 * sin(math.radians(self.yaw + 45))
        pygame.draw.line(surface, color, (self.x - dx, self.y - dy), (self.x + dx, self.y + dy), 2)
        pygame.draw.line(surface, color, (self.x - dy, self.y + dx), (self.x + dy, self.y - dx), 2)
        rotor_r = 3.5
        for sx, sy in [(dx, dy), (-dx, -dy), (dy, -dx), (-dy, dx)]:
            pygame.draw.circle(surface, color, (int(self.x + sx), int(self.y + sy)), rotor_r)
