"""Occupancy grid map for 2-D environment representation.

The :class:`OccupancyGridMap` discretises the continuous 2-D world into a
uniform grid where each cell is labelled as *unexplored*, *unoccupied*,
*occupied* (static), or *dynamic-occupied*.
"""

from __future__ import annotations

import numpy as np
import pygame

from drone2d.config import (
    GRID_DYNAMIC_OCCUPIED,
    GRID_OCCUPIED,
    GRID_UNOCCUPIED,
    GRID_UNEXPLORED,
    COLOR_OCCUPIED,
    COLOR_UNOCCUPIED,
    COLOR_UNEXPLORED,
)


class OccupancyGridMap:
    """Fixed-resolution 2-D occupancy grid.

    Parameters
    ----------
    grid_scale : int
        Side-length (in pixels) of each grid cell.
    dimensions : list[int]
        World size ``[width, height]`` in pixels.
    init_value : int
        Initial cell value (typically :data:`GRID_UNOCCUPIED` for ground-truth
        maps or :data:`GRID_UNEXPLORED` for the drone's perceived map).
    """

    def __init__(
        self,
        grid_scale: int,
        dimensions: list[int],
        init_value: int,
    ) -> None:
        self.dim = dimensions
        self.width = dimensions[0] // grid_scale
        self.height = dimensions[1] // grid_scale
        self.x_scale = grid_scale
        self.y_scale = grid_scale

        self.grid_map = np.full(
            (self.width, self.height), init_value, dtype=np.uint8
        )
        self._dynamic_indices: list[list[int]] = []

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def init_obstacles(self, static_obstacles: list, agents: list) -> None:
        """Mark map edges, static obstacles and initial agent positions.

        Parameters
        ----------
        static_obstacles : list
            Each element is ``[x, y, radius]``.
        agents : list[Agent]
            Dynamic obstacle agents (for initial placement only).
        """
        # Map boundaries
        self.grid_map[0, :] = GRID_OCCUPIED
        self.grid_map[-1, :] = GRID_OCCUPIED
        self.grid_map[:, 0] = GRID_OCCUPIED
        self.grid_map[:, -1] = GRID_OCCUPIED

        for i in range(self.width):
            for j in range(self.height):
                cell_pos = self._cell_center(i, j)
                # Static obstacles
                for obs in static_obstacles:
                    if np.linalg.norm(cell_pos - np.array(obs[:2])) <= obs[2]:
                        self.grid_map[i, j] = GRID_OCCUPIED
                # Dynamic agents
                for agent in agents:
                    dx = cell_pos[0] - agent.position[0]
                    dy = cell_pos[1] - agent.position[1]
                    if dx * dx + dy * dy <= agent.radius ** 2:
                        self.grid_map[i, j] = GRID_DYNAMIC_OCCUPIED
                        self._dynamic_indices.append([i, j])

    # ------------------------------------------------------------------
    # Runtime updates
    # ------------------------------------------------------------------
    def update_dynamic_grid(self, agents: list) -> None:
        """Re-compute dynamic obstacle cells for the current agent positions."""
        # Clear previous dynamic cells
        for idx in self._dynamic_indices:
            self.grid_map[idx[0], idx[1]] = GRID_UNOCCUPIED
        self._dynamic_indices.clear()

        for agent in agents:
            ux = int(agent.radius // self.x_scale)
            uy = int(agent.radius // self.y_scale)
            cx = int(agent.position[0] // self.x_scale)
            cy = int(agent.position[1] // self.y_scale)
            for i in range(max(cx - ux, 0), min(cx + ux + 1, self.width)):
                for j in range(max(cy - uy, 0), min(cy + uy + 1, self.height)):
                    if self.grid_map[i, j] != GRID_OCCUPIED:
                        self.grid_map[i, j] = GRID_DYNAMIC_OCCUPIED
                        self._dynamic_indices.append([i, j])

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_grid(self, x: float, y: float) -> int:
        """Return the cell value at world coordinates ``(x, y)``.

        Out-of-bounds queries return :data:`GRID_OCCUPIED`.
        """
        if x >= self.dim[0] or x < 0 or y >= self.dim[1] or y < 0:
            return GRID_OCCUPIED
        return int(self.grid_map[int(x // self.x_scale), int(y // self.y_scale)])

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self, surface: pygame.Surface) -> None:
        """Draw the grid map onto a pygame *surface*."""
        for i in range(self.width):
            for j in range(self.height):
                cell = self.grid_map[i, j]
                if cell == GRID_OCCUPIED:
                    color = COLOR_OCCUPIED
                elif cell in (GRID_UNOCCUPIED, GRID_DYNAMIC_OCCUPIED):
                    color = COLOR_UNOCCUPIED
                else:
                    color = COLOR_UNEXPLORED
                pygame.draw.rect(
                    surface,
                    color,
                    (
                        self.x_scale * i,
                        self.y_scale * j,
                        self.x_scale,
                        self.y_scale,
                    ),
                )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _cell_center(self, i: int, j: int) -> np.ndarray:
        """Return the world-coordinate centre of cell ``(i, j)``."""
        return np.array(
            [self.x_scale * (i + 0.5), self.y_scale * (j + 0.5)]
        )
