"""drone2d — 2-D Drone Active Perception Simulation.

A modular OpenAI Gym environment for studying navigation and active
perception of MAVs with limited field-of-view in unknown, dynamic
environments.

Quick reference of public API::

    from drone2d.config import SimConfig
    from drone2d.drone import Drone2D
    from drone2d.agent import Agent, rvo_update
    from drone2d.grid_map import OccupancyGridMap
    from drone2d.kalman_filter import KalmanFilter
    from drone2d.trajectory import Trajectory2D, Waypoint2D
    from drone2d.planners import get_planner
    from drone2d.gaze import get_gaze_planner
"""

from drone2d.config import SimConfig
from drone2d.drone import Drone2D
from drone2d.agent import Agent
from drone2d.trajectory import Trajectory2D, Waypoint2D

__all__ = [
    "SimConfig",
    "Drone2D",
    "Agent",
    "Trajectory2D",
    "Waypoint2D",
]
