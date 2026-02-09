"""Rotating gaze policy — constant yaw rotation."""

from __future__ import annotations

from drone2d.config import SimConfig


class Rotating:
    """The drone continuously rotates at maximum yaw speed."""

    def __init__(self, config: SimConfig) -> None:
        self.config = config

    def plan(self, observation: dict) -> float:
        return 1.0
