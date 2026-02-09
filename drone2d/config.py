"""Simulation configuration and shared constants.

This module defines:
- Grid cell types for the occupancy map.
- State machine states for the drone navigation loop.
- Color palette used for pygame rendering.
- :class:`SimConfig` — a dataclass that holds every tunable parameter of the
  simulation and can be constructed from command-line arguments.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Grid cell types
# ---------------------------------------------------------------------------
GRID_UNEXPLORED = 0
GRID_OCCUPIED = 1
GRID_UNOCCUPIED = 2
GRID_DYNAMIC_OCCUPIED = 3

# ---------------------------------------------------------------------------
# State machine states for the navigation loop
# ---------------------------------------------------------------------------
STATE_WAIT_FOR_GOAL = 0
STATE_GOAL_REACHED = 1
STATE_PLANNING = 2
STATE_EXECUTING = 3

# ---------------------------------------------------------------------------
# Rendering colour palette (RGB)
# ---------------------------------------------------------------------------
COLOR_OCCUPIED = (100, 100, 100)
COLOR_UNOCCUPIED = (200, 200, 200)
COLOR_UNEXPLORED = (240, 240, 240)


# ---------------------------------------------------------------------------
# Main simulation configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    """All tuneable parameters for the 2D active-perception simulation.

    Attributes
    ----------
    env : str
        Gym environment identifier.
    render : bool
        If *True* the simulation is visualised with pygame.
    record : bool
        If *True* experiment results are appended to a CSV file.
    record_img : bool
        If *True* screenshots are saved on collision events.
    trained_policy : bool
        Whether to load a pre-trained RL policy for gaze control.
    policy_dir : str
        Path to the trained policy checkpoint.
    dt : float
        Simulation time-step in seconds.
    map_scale : int
        Side-length (in pixels) of each grid cell.
    map_size : list[int]
        Map dimensions ``[width, height]`` in pixels.
    agent_radius : int
        Radius of the dynamic obstacle agents (pixels).
    agent_number : int
        Number of dynamic obstacle agents.
    agent_max_speed : int
        Maximum speed of the agents (pixels / s).
    drone_max_acceleration : int
        Maximum drone acceleration (pixels / s²).
    drone_radius : int
        Drone collision radius (pixels).
    drone_max_speed : int
        Maximum drone speed (pixels / s).
    drone_max_yaw_speed : int
        Maximum yaw rotation speed (degrees / s).
    drone_view_depth : int
        Sensor range (pixels).
    drone_view_range : int
        Field-of-view angle (degrees).
    var_cam : int
        Sensor measurement noise variance.
    motion_profile : str
        Agent motion model: ``'CVM'`` (constant velocity) or ``'RVO'``
        (reciprocal velocity obstacles).
    pillar_number : int
        Number of static circular pillar obstacles.
    map_id : int
        Random seed / map identifier.
    init_position : list[int]
        Drone starting position ``[x, y]``.
    target_list : list
        List of target waypoints ``[[x, y], …]``.
    static_map : str
        Path to a ``.npy`` static obstacle map.
    img_dir : str
        Directory where screenshots are saved.
    max_flight_time : int
        Maximum episode duration in seconds.
    gaze_method : str
        Gaze planning strategy name (see :mod:`drone2d.gaze`).
    planner : str
        Trajectory planner name (see :mod:`drone2d.planners`).
    """

    env: str = "drone-2d-perception-v2"
    render: bool = True
    record: bool = False
    record_img: bool = False
    trained_policy: bool = False
    policy_dir: str = "./trained_policy/lookahead.zip"
    dt: float = 0.1
    map_scale: int = 10
    map_size: List[int] = field(default_factory=lambda: [500, 500])
    agent_radius: int = 10
    agent_number: int = 10
    agent_max_speed: int = 40
    drone_max_acceleration: int = 40
    drone_radius: int = 10
    drone_max_speed: int = 40
    drone_max_yaw_speed: int = 80
    drone_view_depth: int = 80
    drone_view_range: int = 90
    var_cam: int = 0
    motion_profile: str = "CVM"
    pillar_number: int = 0
    map_id: int = 0
    init_position: List[int] = field(default_factory=lambda: [50, 50])
    target_list: List = field(default_factory=lambda: [[50, 460]])
    static_map: str = "maps/empty_map.npy"
    img_dir: str = "./"
    max_flight_time: int = 80
    gaze_method: str = "LookAhead"
    planner: str = "Primitive"

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls, args: list | None = None) -> SimConfig:
        """Build a :class:`SimConfig` from command-line arguments.

        Parameters
        ----------
        args : list[str] | None
            Explicit argument list.  When *None* ``sys.argv`` is used.

        Returns
        -------
        SimConfig
        """
        parser = argparse.ArgumentParser(
            description="2D Drone Active Perception Simulation",
        )
        parser.add_argument("--env", default="drone-2d-perception-v2")
        parser.add_argument(
            "--debug",
            action="store_false",
            help="Debug mode: render enabled, recording disabled",
        )
        parser.add_argument("--record_img", action="store_true")
        parser.add_argument("--trained_policy", action="store_true")
        parser.add_argument(
            "--policy_dir", default="./trained_policy/lookahead.zip"
        )
        parser.add_argument("--dt", type=float, default=0.1)
        parser.add_argument("--map_scale", type=int, default=10)
        parser.add_argument(
            "--map_size", nargs=2, type=int, default=[500, 500]
        )
        parser.add_argument("--agent_radius", type=int, default=10)
        parser.add_argument("--agent_number", type=int, default=10)
        parser.add_argument("--agent_max_speed", type=int, default=40)
        parser.add_argument("--drone_max_acceleration", type=int, default=40)
        parser.add_argument("--drone_radius", type=int, default=10)
        parser.add_argument("--drone_max_speed", type=int, default=40)
        parser.add_argument("--drone_max_yaw_speed", type=int, default=80)
        parser.add_argument("--drone_view_depth", type=int, default=80)
        parser.add_argument("--drone_view_range", type=int, default=90)
        parser.add_argument("--var_cam", type=int, default=0)
        parser.add_argument("--motion_profile", default="CVM")
        parser.add_argument("--pillar_number", type=int, default=0)
        parser.add_argument("--map_id", type=int, default=0)
        parser.add_argument(
            "--init_pos", nargs=2, type=int, default=[50, 50]
        )
        parser.add_argument(
            "--target_list", nargs="+", type=int, default=[[50, 460]]
        )
        parser.add_argument("--static_map", default="maps/empty_map.npy")
        parser.add_argument("--img_dir", default="./")
        parser.add_argument("--max_flight_time", type=int, default=80)
        parser.add_argument("--gaze_method", default="LookAhead")
        parser.add_argument("--planner", default="Primitive")

        parsed = parser.parse_args(args)
        is_debug = parsed.debug

        return cls(
            env=parsed.env,
            render=is_debug,
            record=not is_debug,
            record_img=parsed.record_img,
            trained_policy=parsed.trained_policy,
            policy_dir=parsed.policy_dir,
            dt=parsed.dt,
            map_scale=parsed.map_scale,
            map_size=parsed.map_size,
            agent_radius=parsed.agent_radius,
            agent_number=parsed.agent_number,
            agent_max_speed=parsed.agent_max_speed,
            drone_max_acceleration=parsed.drone_max_acceleration,
            drone_radius=parsed.drone_radius,
            drone_max_speed=parsed.drone_max_speed,
            drone_max_yaw_speed=parsed.drone_max_yaw_speed,
            drone_view_depth=parsed.drone_view_depth,
            drone_view_range=parsed.drone_view_range,
            var_cam=parsed.var_cam,
            motion_profile=parsed.motion_profile,
            pillar_number=parsed.pillar_number,
            map_id=parsed.map_id,
            init_position=parsed.init_pos,
            target_list=parsed.target_list,
            static_map=parsed.static_map,
            img_dir=parsed.img_dir,
            max_flight_time=parsed.max_flight_time,
            gaze_method=parsed.gaze_method,
            planner=parsed.planner,
        )
