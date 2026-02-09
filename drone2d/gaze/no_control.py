"""No gaze control — the drone has 360-degree vision."""

from __future__ import annotations

from drone2d.config import SimConfig


class NoControl:
    """Dummy gaze policy that outputs zero yaw command."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config

    def plan(self, state: dict) -> float:
        return 0.0
