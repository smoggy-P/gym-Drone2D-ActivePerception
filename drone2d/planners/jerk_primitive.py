"""Jerk-optimal motion primitive planner.

Generates minimum-jerk polynomial trajectories (5th-order) towards
sampled heading directions and selects the one with lowest cost.
"""

from __future__ import annotations

import math
from math import atan2, radians

import numpy as np
from numpy.linalg import norm

from drone2d.config import SimConfig
from drone2d.drone import Drone2D
from drone2d.trajectory import Trajectory2D
from drone2d.planners.base import Planner


class JerkPrimitive(Planner):
    """Jerk-optimal motion-primitive planner.

    At each step a set of candidate heading directions is evaluated and the
    collision-free primitive with the lowest directional cost is executed.

    Parameters
    ----------
    drone : Drone2D
        Reference drone.
    config : SimConfig
        Simulation parameters.
    """

    def __init__(self, drone: Drone2D, config: SimConfig) -> None:
        super().__init__(drone, config)
        self._theta_range = np.arange(0, 360, 5)
        self._d = 30  # primitive arc-length
        self._k1 = 1.0  # heading cost weight

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def plan(self, drone: Drone2D, update_dt: float) -> bool:
        start_pos = np.array([drone.x, drone.y])
        start_vel = drone.velocity
        start_acc = drone.acceleration
        occ = drone.map
        v_max = self.config.drone_max_speed

        # Target heading
        delta_p = self.target[:2] - start_pos
        phi_h = math.degrees(atan2(delta_p[1], delta_p[0]))

        # Cost per candidate heading
        costs = np.zeros((len(self._theta_range), 2))
        for i, theta in enumerate(self._theta_range):
            diff = abs(theta % 360 - phi_h % 360)
            diff = min(diff, 360 - diff)
            costs[i] = [self._k1 * diff ** 2, theta]
        costs = costs[costs[:, 0].argsort()]

        collision = True
        for seq in range(len(self._theta_range)):
            ps, vs, accs, ts, *_ = self._generate_primitive(
                start_pos, start_vel, start_acc, costs[seq, 1], v_max, update_dt,
            )
            collision = False
            for t, pos in zip(ts, ps):
                if not self.is_free(pos, t, occ, drone.trackers):
                    collision = True
                    break
            if not collision:
                break

        if collision:
            return False

        self.trajectory.positions.append(ps[0])
        self.trajectory.velocities.append(vs[0])
        self.trajectory.accelerations.append(accs[0])
        return True

    def replan_check(self, drone: Drone2D) -> tuple[bool, np.ndarray]:
        occ = drone.map.grid_map
        swept = np.zeros_like(occ)
        for i, pos in enumerate(self.trajectory.positions):
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _generate_primitive(self, p0, v0, a0, theta_h, v_max, dt):
        """Generate a 5th-order jerk-optimal trajectory primitive."""
        dx = self._d * np.cos(radians(theta_h))
        dy = self._d * np.sin(radians(theta_h))
        pf = p0 + np.array([dx, dy])

        direction = self.target[:2] - pf
        dir_norm = norm(direction)
        vf = (0.5 * v_max / dir_norm) * direction if dir_norm > 0 else np.zeros(2)
        af = np.zeros(2)

        T = max(1.2 * norm([dx, dy]) / norm(v_max), 0.5)
        n_steps = int(np.floor(T / dt))
        t = np.arange(dt, n_steps * dt + dt, dt)

        p = np.zeros((n_steps, 2))
        v = np.zeros((n_steps, 2))
        a = np.zeros((n_steps, 2))

        for ax in range(2):
            da = af[ax] - a0[ax]
            dv = vf[ax] - v0[ax] - a0[ax] * T
            dp = pf[ax] - p0[ax] - v0[ax] * T - 0.5 * a0[ax] * T ** 2
            alpha = da * 60 / T ** 3 - dv * 360 / T ** 4 + dp * 720 / T ** 5
            beta = -da * 24 / T ** 2 + dv * 168 / T ** 3 - dp * 360 / T ** 4
            gamma = da * 3 / T - dv * 24 / T ** 2 + dp * 60 / T ** 3
            for j in range(n_steps):
                tt = t[j]
                p[j, ax] = (
                    alpha / 120 * tt ** 5
                    + beta / 24 * tt ** 4
                    + gamma / 6 * tt ** 3
                    + a0[ax] / 2 * tt ** 2
                    + v0[ax] * tt
                    + p0[ax]
                )
                v[j, ax] = (
                    alpha / 24 * tt ** 4
                    + beta / 6 * tt ** 3
                    + gamma / 2 * tt ** 2
                    + a0[ax] * tt
                    + v0[ax]
                )
                a[j, ax] = (
                    alpha / 6 * tt ** 3
                    + beta / 2 * tt ** 2
                    + gamma * tt
                    + a0[ax]
                )
        return p, v, a, t, pf, vf, af
