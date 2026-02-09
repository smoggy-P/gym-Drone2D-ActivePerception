"""Model Predictive Control (MPC) trajectory planner.

.. note::
   This planner requires the proprietary ``forcespro`` solver library.
   If ``forcespro`` is not installed, importing this module will succeed but
   instantiating :class:`MPC` will raise a clear error message.
"""

from __future__ import annotations

from math import radians

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy.linalg import norm
from sklearn.cluster import DBSCAN

from drone2d.config import GRID_OCCUPIED, SimConfig
from drone2d.drone import Drone2D
from drone2d.trajectory import Trajectory2D
from drone2d.planners.base import Planner

_HAS_FORCESPRO = False
try:
    import forcespro  # noqa: F401
    _HAS_FORCESPRO = True
except ImportError:
    pass


class MPC(Planner):
    """MPC-based trajectory planner using FORCESPRO.

    Parameters
    ----------
    drone : Drone2D
        Reference drone.
    config : SimConfig
        Simulation parameters.

    Raises
    ------
    RuntimeError
        If ``forcespro`` is not installed.
    """

    def __init__(self, drone: Drone2D, config: SimConfig) -> None:
        super().__init__(drone, config)
        if not _HAS_FORCESPRO:
            raise RuntimeError(
                "The MPC planner requires 'forcespro'. "
                "Please install it from https://www.embotech.com/FORCES-Pro"
            )
        self._solver = forcespro.nlp.Solver.from_directory("./mpc/MPC_SOLVER/")
        self.N = 25
        self.future_trajectory = Trajectory2D()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def plan(self, drone: Drone2D, update_dt: float) -> bool:
        if len(self.trajectory) != 0:
            return True

        ms = self.config.map_scale
        w = int(self.config.map_size[0] // ms)
        h = int(self.config.map_size[1] // ms)

        x_grid = np.arange(w).reshape(-1, 1) * ms
        y_grid = np.arange(h).reshape(1, -1) * ms
        local_obs = np.where(
            np.logical_and(
                (drone.x - x_grid) ** 2 + (drone.y - y_grid) ** 2 <= 100 ** 2,
                drone.map.grid_map == GRID_OCCUPIED,
            ),
            1, 0,
        )
        positions, widths, heights, angles = _binary_image_clustering(
            local_obs, 1.5, 1, np.array([drone.x, drone.y]),
        )

        obs_list = []
        for tracker in drone.trackers:
            if tracker.active:
                obs_list.append([
                    *tracker.mu_upds[-1][:2, 0],
                    tracker.radius + 10,
                    tracker.radius + 10,
                    0,
                    *tracker.mu_upds[-1][2:, 0],
                ])
        for i in range(len(positions)):
            obs_list.append([
                *positions[i], widths[i], heights[i], radians(angles[i]), 0, 0,
            ])

        problem = {"xinit": np.array([drone.x, drone.y, *drone.velocity])}
        all_params = []
        for i in range(self.N):
            obs_i = []
            for obs in obs_list:
                o = list(obs)
                o[0] += o[5] * update_dt * i
                o[1] += o[6] * update_dt * i
                obs_i.append(o)
            obs_i.sort(key=lambda o: (o[0] - drone.x) ** 2 + (o[1] - drone.y) ** 2)
            for j in range(5):
                all_params.extend(obs_i[j][:5] if j < len(obs_i) else [0] * 5)
            all_params.extend([
                self.config.drone_max_speed,
                self.target[0],
                self.target[1],
            ])

        problem["all_parameters"] = np.array(all_params)
        solverout, exitflag, _ = self._solver.solve(problem)

        if exitflag != 1:
            self.trajectory = Trajectory2D()
            self.future_trajectory = Trajectory2D()
            return False

        self.trajectory = Trajectory2D()
        self.future_trajectory = Trajectory2D()

        self.trajectory.positions.append(np.array([solverout["x02"][3], solverout["x02"][4]]))
        self.trajectory.velocities.append(np.array([solverout["x02"][5], solverout["x02"][6]]))
        self.trajectory.accelerations.append(np.zeros(2))

        for z in solverout.values():
            self.future_trajectory.positions.append(np.array([z[3], z[4]]))
            self.future_trajectory.velocities.append(np.array([z[5], z[6]]))
            self.future_trajectory.accelerations.append(np.zeros(2))

        return True

    # ------------------------------------------------------------------
    def replan_check(self, drone: Drone2D) -> tuple[bool, np.ndarray]:
        occ = drone.map.grid_map
        swept = np.zeros_like(occ)
        for i, pos in enumerate(self.future_trajectory.positions):
            gi = int(pos[0] // self.config.map_scale)
            gj = int(pos[1] // self.config.map_scale)
            swept[gi, gj] = i * self.config.dt
            for tracker in drone.trackers:
                if tracker.active:
                    if norm(tracker.estimate_pos(i * self.config.dt) - pos) <= self.config.drone_radius + tracker.radius:
                        self.trajectory.clear()
                        return True, swept
        if np.sum(np.where(occ == 1, 1, 0) * swept) > 0:
            self.trajectory.clear()
            return True, swept
        return False, swept


# ======================================================================
# Helpers
# ======================================================================

def _binary_image_clustering(image, eps, min_samples, start_pos):
    """Cluster occupied cells into ellipsoidal obstacles via DBSCAN."""
    img = image.copy()
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 0
    indices = np.array(np.where(img == 1)).T
    if indices.shape[0] == 0:
        return [], [], [], []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(indices)
    positions, widths, heights, angles = [], [], [], []

    for k in set(labels):
        xy = indices[labels == k]
        if xy.shape[0] == 1:
            w, h, a = 10, 10, 0
            pos = 5 + 10 * xy[0]
        else:
            cov = np.cov(xy, rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            a = np.degrees(np.arctan2(*evecs[:, 0][::-1]))
            w, h = 25 * np.sqrt(evals)
            pos = 5 + 10 * xy.mean(axis=0)
        positions.append(pos)
        widths.append(w)
        heights.append(h)
        angles.append(a)

    return positions, widths, heights, angles
