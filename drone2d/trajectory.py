"""Trajectory and waypoint data structures.

Provides lightweight containers used by the trajectory planners to store
planned position, velocity and acceleration profiles.
"""

from __future__ import annotations

import numpy as np


class Waypoint2D:
    """A single 2-D waypoint with position and velocity."""

    __slots__ = ("position", "velocity")

    def __init__(
        self,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
    ) -> None:
        self.position = position if position is not None else np.zeros(2)
        self.velocity = velocity if velocity is not None else np.zeros(2)


class Trajectory2D:
    """An ordered sequence of waypoints forming a 2-D trajectory.

    Each waypoint is stored as three parallel lists of NumPy arrays:
    ``positions``, ``velocities`` and ``accelerations``.
    """

    def __init__(self) -> None:
        self.positions: list[np.ndarray] = []
        self.velocities: list[np.ndarray] = []
        self.accelerations: list[np.ndarray] = []

    # ------------------------------------------------------------------
    # Sequence-like helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.positions)

    def pop_front(self) -> None:
        """Remove the first waypoint from the trajectory."""
        self.positions.pop(0)
        self.velocities.pop(0)
        self.accelerations.pop(0)

    def clear(self) -> None:
        """Remove all waypoints."""
        self.positions.clear()
        self.velocities.clear()
        self.accelerations.clear()
