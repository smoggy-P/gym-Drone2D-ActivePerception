"""Experiment runner — execute one simulation episode and record results.

The :class:`Experiment` class wires together an environment and a gaze policy,
runs the episode loop, and optionally appends the outcome to a CSV file.
"""

from __future__ import annotations

import os

import gym
import numpy as np
import pandas as pd

import drone2d.envs  # noqa: F401  — triggers Gym registration
from drone2d.config import STATE_GOAL_REACHED, SimConfig
from drone2d.gaze import get_gaze_planner


# CSV column names for experiment results
_RESULT_COLUMNS = [
    "Method", "Planner", "Motion Profile", "Map ID",
    "Agent size", "Number of agents", "Number of pillars",
    "Agent speed", "Drone speed", "Depth variance",
    "Initial position", "Target position",
    "Flight time", "Grid discovered", "Agent tracked",
    "Agent tracked time", "Success",
    "Static Collision", "Dynamic Collision",
    "Freezing", "Dead Lock", "state machine",
]


def _append_csv(path: str, row: tuple) -> None:
    """Append a single result row to *path*."""
    df = pd.read_csv(path, index_col=False)
    df.loc[len(df)] = row
    df.to_csv(path, index=False)


class Experiment:
    """Single-episode experiment runner.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration.
    result_dir : str
        Path to the CSV results file.  Created automatically if it does not
        exist.
    """

    def __init__(self, config: SimConfig, result_dir: str) -> None:
        if config.gaze_method == "NoControl":
            config.drone_view_range = 360

        self.config = config
        self.env = gym.make(config.env, config=config)
        self.dt = config.dt
        self.result_dir = result_dir

        # Instantiate the gaze planner properly
        gaze_cls = get_gaze_planner(config.gaze_method)
        self.gaze = gaze_cls(config)

        # Create CSV header if needed
        if config.record and not os.path.isfile(result_dir):
            pd.DataFrame(columns=_RESULT_COLUMNS).to_csv(result_dir, index=False)

    def run(self) -> None:
        """Execute the episode until termination."""
        self.env.reset()
        done = False

        while not done:
            action = self.gaze.plan(self.env.info)
            _, _, done, info = self.env.step(action)

            if done and self.config.record:
                tracking_time = np.array([
                    len(t.ts) * 0.1 for t in info["tracker_buffer"]
                ]).sum()
                n_tracked = len(info["tracker_buffer"])
                grid_map = info["drone"].map.grid_map
                grid_discovered = (
                    grid_map.shape[0] * grid_map.shape[1]
                    - np.sum(grid_map == 0)
                )

                row = (
                    self.config.gaze_method,
                    self.config.planner,
                    self.config.motion_profile,
                    self.config.map_id,
                    self.config.agent_radius,
                    self.config.agent_number,
                    self.config.pillar_number,
                    self.config.agent_max_speed,
                    self.config.drone_max_speed,
                    self.config.var_cam,
                    self.config.init_position,
                    self.config.target_list[0] if self.config.target_list else None,
                    info["flight_time"],
                    grid_discovered,
                    n_tracked,
                    tracking_time / max(n_tracked, 1),
                    int(info["state_machine"] == STATE_GOAL_REACHED),
                    int(info["collision_flag"] == 1),
                    int(info["collision_flag"] == 2),
                    info["freezing_flag"],
                    info["dead_lock_flag"],
                    info["state_machine"],
                )
                _append_csv(self.result_dir, row)

            if self.config.render:
                self.env.render()
