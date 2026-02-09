"""Dynamic obstacle agents and Reciprocal Velocity Obstacles (RVO).

This module provides:

* :class:`Agent` — a disk-shaped dynamic obstacle with position, velocity and
  preferred-velocity.
* :func:`rvo_update` — one RVO collision-avoidance step that mutates agent
  velocities in-place.
"""

from __future__ import annotations

from math import atan2, asin, cos, sin, pi

import numpy as np
from numpy.linalg import norm


class Agent:
    """A disk-shaped dynamic obstacle that moves in 2-D.

    Parameters
    ----------
    position : tuple | np.ndarray
        Initial ``(x, y)`` position.
    velocity : tuple | np.ndarray
        Initial ``(vx, vy)`` velocity.
    radius : float
        Collision radius in pixels.
    max_speed : float
        Maximum speed (pixels / s).
    pref_velocity : tuple | np.ndarray
        Desired / preferred velocity.
    group_id : int
        Identifier for grouped (shaped-map) obstacles.
    """

    def __init__(
        self,
        position,
        velocity,
        radius: float,
        max_speed: float,
        pref_velocity,
        group_id: int = 0,
    ) -> None:
        self.position = np.asarray(position, dtype=float)
        self.velocity = np.asarray(velocity, dtype=float)
        self.radius = float(radius)
        self.max_speed = max_speed
        self.pref_velocity = np.asarray(pref_velocity, dtype=float)
        self.group_id = group_id

    def step(
        self,
        edge_x: float,
        edge_y: float,
        map_width: float,
        map_height: float,
        dt: float,
    ) -> None:
        """Advance the agent by one time-step.

        The agent bounces off map boundaries and rotates its preferred
        velocity when it gets stuck.
        """
        new_pos = self.position + self.velocity * dt

        # Rotate preferred velocity when stuck
        if norm(self.velocity) <= 5:
            rot = np.array(
                [
                    [cos(pi / 6), sin(-pi / 6)],
                    [sin(pi / 6), cos(pi / 6)],
                ]
            )
            self.pref_velocity = (rot @ self.pref_velocity.reshape(-1, 1)).flatten()

        # Bounce off map boundaries
        if new_pos[0] < edge_x + self.radius:
            self.pref_velocity[0] = abs(self.pref_velocity[0])
        elif new_pos[0] > map_width - edge_x - self.radius:
            self.pref_velocity[0] = -abs(self.pref_velocity[0])

        if new_pos[1] < edge_y + self.radius:
            self.pref_velocity[1] = abs(self.pref_velocity[1])
        elif new_pos[1] > map_height - edge_y - self.radius:
            self.pref_velocity[1] = -abs(self.pref_velocity[1])

        self.position = self.position + self.velocity * dt


# ======================================================================
# Reciprocal Velocity Obstacles  (RVO)
# ======================================================================

def _in_between(theta_right: float, theta_dif: float, theta_left: float) -> bool:
    """Check whether *theta_dif* lies between *theta_right* and *theta_left*."""
    if abs(theta_right - theta_left) <= pi:
        return theta_right <= theta_dif <= theta_left

    if theta_left < 0 < theta_right:
        theta_left += 2 * pi
        if theta_dif < 0:
            theta_dif += 2 * pi
        return theta_right <= theta_dif <= theta_left

    if theta_left > 0 > theta_right:
        theta_right += 2 * pi
        if theta_dif < 0:
            theta_dif += 2 * pi
        return theta_left <= theta_dif <= theta_right

    return False


def _intersect(
    pA: np.ndarray,
    vA: np.ndarray,
    rvo_ba_all: list,
) -> np.ndarray:
    """Find the best collision-free velocity closest to *vA*."""
    norm_v = norm(vA)
    suitable: list[np.ndarray] = []
    unsuitable: list[np.ndarray] = []

    # Sample candidate velocities
    for theta in np.arange(0, 2 * pi, 0.2):
        for rad in np.arange(0.02, norm_v + 0.02, norm_v / 5.0):
            new_v = np.array([rad * cos(theta), rad * sin(theta)])
            suit = True
            for rvo in rvo_ba_all:
                td = atan2(new_v[1] + pA[1] - rvo[0][1], new_v[0] + pA[0] - rvo[0][0])
                tr = atan2(rvo[2][1], rvo[2][0])
                tl = atan2(rvo[1][1], rvo[1][0])
                if _in_between(tr, td, tl):
                    suit = False
                    break
            (suitable if suit else unsuitable).append(new_v)

    # Also test the desired velocity itself
    new_v = vA.copy()
    suit = True
    for rvo in rvo_ba_all:
        td = atan2(new_v[1] + pA[1] - rvo[0][1], new_v[0] + pA[0] - rvo[0][0])
        tr = atan2(rvo[2][1], rvo[2][0])
        tl = atan2(rvo[1][1], rvo[1][0])
        if _in_between(tr, td, tl):
            suit = False
            break
    (suitable if suit else unsuitable).append(new_v)

    if suitable:
        return min(suitable, key=lambda v: norm(v - vA))

    # Fallback: pick the unsuitable velocity with the best trade-off
    tc_map: dict[tuple, float] = {}
    for uv in unsuitable:
        tc_list: list[float] = []
        for rvo in rvo_ba_all:
            dif = np.array([uv[0] + pA[0] - rvo[0][0], uv[1] + pA[1] - rvo[0][1]])
            td = atan2(dif[1], dif[0])
            tr = atan2(rvo[2][1], rvo[2][0])
            tl = atan2(rvo[1][1], rvo[1][0])
            if _in_between(tr, td, tl):
                small = abs(td - 0.5 * (tl + tr))
                rad_val = rvo[4]
                dist_val = rvo[3]
                if abs(dist_val * sin(small)) >= rad_val:
                    rad_val = abs(dist_val * sin(small))
                big = asin(abs(dist_val * sin(small)) / rad_val)
                dist_tg = max(abs(dist_val * cos(small)) - abs(rad_val * cos(big)), 0)
                tc_list.append(dist_tg / norm(dif))
        tc_map[tuple(uv)] = min(tc_list) + 0.001

    wt = 0.2
    return min(unsuitable, key=lambda v: (wt / tc_map[tuple(v)]) + norm(v - vA))


def rvo_update(
    agents: list[Agent],
    circular_obstacles: list,
) -> bool:
    """Run one RVO collision-avoidance step for all *agents*.

    Each agent's ``velocity`` is updated **in-place**.

    Parameters
    ----------
    agents : list[Agent]
        The dynamic agents.
    circular_obstacles : list
        Static circular obstacles, each ``[x, y, radius]``.

    Returns
    -------
    bool
        *True* on success, *False* if a numerical error occurred.
    """
    positions = [a.position for a in agents]
    desired = [a.pref_velocity for a in agents]
    current = [a.velocity for a in agents]
    rob_rad = agents[0].radius + 0.01

    for i in range(len(positions)):
        try:
            vA = current[i].copy()
            pA = positions[i].copy()
            rvo_all: list = []

            # --- Other agents -------------------------------------------
            for j in range(len(positions)):
                if i == j:
                    continue
                vB = current[j].copy()
                pB = positions[j].copy()
                transl = np.array([
                    pA[0] + 0.5 * (vB[0] + vA[0]),
                    pA[1] + 0.5 * (vB[1] + vA[1]),
                ])
                dist = max(norm(pA - pB), 2 * rob_rad)
                theta = atan2(pB[1] - pA[1], pB[0] - pA[0])
                half_angle = asin(2 * rob_rad / dist)
                rvo_all.append([
                    transl,
                    [cos(theta + half_angle), sin(theta + half_angle)],
                    [cos(theta - half_angle), sin(theta - half_angle)],
                    dist,
                    2 * rob_rad,
                ])

            # --- Static circular obstacles ------------------------------
            for hole in circular_obstacles:
                pB = np.array(hole[:2])
                transl = np.array([pA[0], pA[1]])
                dist = norm(pA - pB)
                theta = atan2(pB[1] - pA[1], pB[0] - pA[0])
                rad = hole[2] * 1.5  # over-approximation
                dist = max(dist, rad + rob_rad)
                half_angle = asin((rad + rob_rad) / dist)
                rvo_all.append([
                    transl,
                    [cos(theta + half_angle), sin(theta + half_angle)],
                    [cos(theta - half_angle), sin(theta - half_angle)],
                    dist,
                    rad + rob_rad,
                ])
        except Exception:
            return False

        agents[i].velocity = _intersect(positions[i], desired[i], rvo_all)

    return True
