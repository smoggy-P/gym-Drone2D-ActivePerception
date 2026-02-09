"""Motion-primitive-based trajectory planner.

Uses an A*-like search over polynomial motion primitives to find a
collision-free trajectory from the current drone state to the goal.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from drone2d.config import SimConfig
from drone2d.drone import Drone2D
from drone2d.trajectory import Trajectory2D, Waypoint2D
from drone2d.planners.base import Planner


class _PrimitiveNode:
    """Search node for the motion-primitive graph."""

    __slots__ = (
        "position", "velocity", "cost", "parent_index",
        "coeff", "itr", "total_cost", "index",
    )

    def __init__(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        cost: float,
        target: np.ndarray,
        parent_index,
        coeff,
        itr: int,
    ) -> None:
        self.position = pos
        self.velocity = vel
        self.cost = cost
        self.parent_index = parent_index
        self.coeff = coeff
        self.itr = itr
        self.total_cost = cost + 0.5 * norm(pos - target) + 0.1 * norm(vel)
        self.index = (
            round(pos[0]) // 10,
            round(pos[1]) // 10,
            round(vel[0]),
            round(vel[1]),
        )

    def __lt__(self, other: _PrimitiveNode) -> bool:
        return self.total_cost < other.total_cost


def _waypoint_from_traj(coeff: np.ndarray, t: float) -> Waypoint2D:
    """Evaluate a polynomial trajectory segment at time *t*."""
    wp = Waypoint2D()
    wp.position = np.around(np.array([1, t, t ** 2]) @ coeff.T)
    wp.velocity = np.array([1, 2 * t]) @ coeff[:, 1:].T
    return wp


class Primitive(Planner):
    """Graph-search planner using 2nd-order polynomial motion primitives.

    Parameters
    ----------
    drone : Drone2D
        Reference drone (initial state).
    config : SimConfig
        Simulation parameters.
    """

    def __init__(self, drone: Drone2D, config: SimConfig) -> None:
        super().__init__(drone, config)

        if config.drone_max_speed <= 40:
            step = 0.4 * config.drone_max_speed - 5
        else:
            step = 4
        self.u_space = np.arange(
            -config.drone_max_acceleration, config.drone_max_acceleration, step,
        )
        self._dt = 2  # primitive duration (seconds)
        self._sample_num = config.drone_max_speed * self._dt // config.map_scale
        self._search_threshold = 10

    # ------------------------------------------------------------------
    def plan(self, drone: Drone2D, update_dt: float) -> bool:
        if len(self.trajectory) != 0:
            return True

        start_pos = np.array([drone.x, drone.y])
        start_vel = drone.velocity
        occ = drone.map

        self.trajectory = Trajectory2D()
        start_node = _PrimitiveNode(
            pos=start_pos, vel=start_vel, cost=0,
            target=self.target[:2], parent_index=-1, coeff=None, itr=0,
        )

        open_set: dict = {start_node.index: start_node}
        closed_set: dict = {}
        itr = 0

        while True:
            itr += 1
            if not open_set or itr >= 100:
                return False

            c_id = min(open_set, key=lambda o: open_set[o].total_cost)
            current = open_set.pop(c_id)

            if norm(current.position - self.target[:2]) <= self._search_threshold:
                goal_node = current
                break

            closed_set[c_id] = current

            for x_acc in self.u_space:
                for y_acc in self.u_space:
                    vel_mat = np.array([
                        [current.velocity[0], current.velocity[1]],
                        [x_acc / 2, y_acc / 2],
                    ])
                    end_vel = np.array([1, 2 * self._dt]) @ vel_mat
                    if norm(end_vel) >= self.config.drone_max_speed:
                        continue

                    coeff = np.array([
                        [current.position[0], current.velocity[0], x_acc / 2],
                        [current.position[1], current.velocity[1], y_acc / 2],
                    ])

                    # Collision check along the primitive
                    free = True
                    for t in np.arange(0, self._dt, self._dt / self._sample_num):
                        pos = np.around(np.array([1, t, t ** 2]) @ coeff.T)
                        if not self.is_free(pos, t + current.itr * self._dt, occ, drone.trackers):
                            free = False
                            break
                    if not free:
                        continue

                    pos_mat = np.array([
                        [current.position[0], current.position[1]],
                        [current.velocity[0], current.velocity[1]],
                        [x_acc / 2, y_acc / 2],
                    ])
                    end_pos = np.around(np.array([1, self._dt, self._dt ** 2]) @ pos_mat)

                    succ = _PrimitiveNode(
                        pos=end_pos, vel=end_vel,
                        cost=current.cost + (x_acc ** 2 + y_acc ** 2) / 100 + 10,
                        target=self.target[:2],
                        parent_index=current.index,
                        coeff=coeff,
                        itr=current.itr + 1,
                    )

                    if succ.index in closed_set:
                        continue
                    if succ.index not in open_set or open_set[succ.index].cost > succ.cost:
                        open_set[succ.index] = succ

        # Back-track and build trajectory
        cur = goal_node
        while cur is not start_node:
            ts = np.arange(self._dt, 0, -update_dt)
            self.trajectory.positions.extend(
                [_waypoint_from_traj(cur.coeff, t).position for t in ts]
            )
            self.trajectory.velocities.extend(
                [_waypoint_from_traj(cur.coeff, t).velocity for t in ts]
            )
            self.trajectory.accelerations.extend(
                [np.zeros(2) for _ in ts]
            )
            cur = closed_set[cur.parent_index]

        self.trajectory.positions.reverse()
        self.trajectory.velocities.reverse()
        return True

    # ------------------------------------------------------------------
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
