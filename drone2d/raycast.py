"""2-D ray-casting sensor simulation.

:class:`Raycast` mimics a limited field-of-view depth sensor by casting rays
from the drone position, discovering grid cells and detecting agents within
the sensor cone.
"""

from __future__ import annotations

import math
from math import radians, tan, ceil, cos, sin, pi

import numpy as np
import torch

from drone2d.config import GRID_OCCUPIED, GRID_UNOCCUPIED


class Raycast:
    """Fan-shaped ray-casting sensor.

    Parameters
    ----------
    plane_size : tuple[int, int]
        Projection plane ``(width, height)`` — typically the map size.
    fov_deg : float
        Field-of-view in degrees.
    depth : float
        Maximum sensor range in pixels.
    noise_sigma : float
        Standard deviation of measurement noise added to detections.
    """

    STRIP_WIDTH = 10  # pixel width per ray strip

    def __init__(
        self,
        plane_size: tuple[int, int],
        fov_deg: float,
        depth: float,
        noise_sigma: float,
    ) -> None:
        self.fov = radians(fov_deg)
        self.depth = depth
        self.sigma = noise_sigma

        pw, ph = plane_size
        self._center_x = pw // 2
        self._dist_to_plane = self._center_x / tan(self.fov / 2)
        self._rays_number = ceil(pw / self.STRIP_WIDTH)
        self._rays_angle = self.fov / pw
        self._half_rays = self._rays_number // 2

        self._rad90 = radians(90)
        self._rad270 = radians(270)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def cast_rays(self, player, gt_grid_map, agents: list):
        """Cast all rays and return detection results.

        Parameters
        ----------
        player
            The :class:`~drone2d.drone.Drone2D` instance.
        gt_grid_map : OccupancyGridMap
            Ground-truth occupancy grid.
        agents : list[Agent]
            Dynamic obstacle agents.

        Returns
        -------
        rays : list[dict]
            Per-ray results (hit coordinates, wall flag, hit list).
        newly_tracked : int
            Number of agents detected for the first time in this frame.
        measurements : list[np.ndarray | None]
            Per-agent noisy position measurement or *None*.
        """
        player_angle = 2 * pi - radians(player.yaw)
        rays = [
            self._cast_single_ray(
                player,
                player_angle,
                -self.fov / 2 + self.fov / self._rays_number * i,
                gt_grid_map,
                agents,
            )
            for i in range(self._rays_number)
        ]

        hit_list = torch.zeros(len(agents), dtype=torch.int8)
        for ray in rays:
            hit_list = hit_list | ray["hit_list"]

        newly_tracked = 0
        measurements: list[np.ndarray | None] = [None] * len(agents)
        for i, detected in enumerate(hit_list):
            if detected:
                measurements[i] = agents[i].position + self.sigma * np.random.randn(2)
                if not player.trackers[i].active:
                    newly_tracked += 1

        return rays, newly_tracked, measurements

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _positive_angle(angle: float) -> float:
        angle = math.copysign(abs(angle) % (2 * pi), angle)
        if angle < 0:
            angle += 2 * pi
        return angle

    def _cast_single_ray(self, player, player_angle, ray_angle, gt_map, agents):
        """Cast a single ray and return hit information."""
        x_step_size = gt_map.x_scale - 1
        y_step_size = gt_map.y_scale - 1

        ray_angle = self._positive_angle(player_angle + ray_angle)

        x_hit, y_hit = -1.0, -1.0
        wall_hit = 0
        hit_list = torch.zeros(len(agents), dtype=torch.int8)

        faced_right = ray_angle < self._rad90 or ray_angle > self._rad270
        faced_up = ray_angle > pi

        slope = tan(ray_angle)
        x, y = float(player.x), float(player.y)

        if abs(slope) > 1:
            slope = 1.0 / slope
            y_step = -y_step_size if faced_up else y_step_size
            x_step = y_step * slope
        else:
            x_step = x_step_size if faced_right else -x_step_size
            y_step = x_step * slope

        while 0 < x < gt_map.dim[0] and 0 < y < gt_map.dim[1]:
            gi = int(x // gt_map.x_scale)
            gj = int(y // gt_map.y_scale)

            # Check agent hits
            for k, agent in enumerate(agents):
                if (agent.position[0] - x) ** 2 + (agent.position[1] - y) ** 2 <= agent.radius ** 2:
                    x_hit, y_hit = x, y
                    hit_list[k] = 1
            if x_hit != -1:
                break

            wall = gt_map.grid_map[gi, gj]
            dist_sq = (x - player.x) ** 2 + (y - player.y) ** 2
            if wall == 1 or dist_sq >= self.depth ** 2:
                x_hit, y_hit = x, y
                wall_hit = wall
                if wall == GRID_OCCUPIED:
                    player.map.grid_map[gi, gj] = GRID_OCCUPIED
                break
            else:
                player.map.grid_map[gi, gj] = GRID_UNOCCUPIED

            x += x_step
            y += y_step

        return {"coords": (x_hit, y_hit), "wall": wall_hit, "hit_list": hit_list}
