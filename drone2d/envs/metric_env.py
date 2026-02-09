"""Metric-calculation environment (no perception module).

:class:`MetricEnv` is a variant of :class:`~drone2d.envs.drone_env.Drone2DEnv`
that omits the ray-casting perception pipeline.  It is used to compute
difficulty metrics (survivability, traversibility, etc.) where sensor
modelling is not required.
"""

from __future__ import annotations

import random
from itertools import product as iter_product

import gym
import numpy as np
import pygame
from numpy import array, pi, cos, sin
from numpy.linalg import norm

from drone2d.agent import Agent, rvo_update
from drone2d.config import (
    STATE_EXECUTING,
    STATE_GOAL_REACHED,
    STATE_PLANNING,
    STATE_WAIT_FOR_GOAL,
    SimConfig,
)
from drone2d.drone import Drone2D
from drone2d.grid_map import OccupancyGridMap
from drone2d.planners import get_planner
from drone2d.rendering import draw_covariance_ellipse


class MetricEnv(gym.Env):
    """Simplified env for difficulty-metric computation (no perception).

    Parameters
    ----------
    config : SimConfig
        Full simulation configuration.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, config: SimConfig) -> None:
        super().__init__()
        np.seterr(divide="ignore", invalid="ignore")
        random.seed(config.map_id)
        np.random.seed(config.map_id)

        self.config = config
        self.dt = config.dt
        self.steps = 0
        self.max_steps = config.max_flight_time / config.dt
        self.tracked_agent = 0
        self.tracker_buffer: list = []

        # Pygame
        self._screen = None
        self._clock = None
        if config.render:
            pygame.init()
            self._screen = pygame.display.set_mode(config.map_size)
            self._clock = pygame.time.Clock()

        self.target_list = list(config.target_list)

        self.drone = Drone2D(
            config.init_position[0], config.init_position[1],
            init_yaw=-90, dt=self.dt, config=config,
        )

        self.obstacles: list = []
        self.agents: list[Agent] = []
        self._init_obstacles()

        self.map_gt = OccupancyGridMap(config.map_scale, config.map_size, 2)
        self.map_gt.init_obstacles(self.obstacles, self.agents)

        planner_cls = get_planner(config.planner)
        self.planner = planner_cls(self.drone, config)
        self.state_machine = STATE_WAIT_FOR_GOAL
        self.fail_count = 0

        self.action_space = gym.spaces.Box(
            np.array([-1.0]), np.array([1.0]), shape=(1,),
        )
        lm = 4 * (config.drone_view_depth // config.map_scale) + 1
        self.observation_space = gym.spaces.Dict({
            "yaw_angle": gym.spaces.Box(
                low=np.zeros(1, dtype=np.float32),
                high=np.full(1, 360, dtype=np.float32),
            ),
            "local_map": gym.spaces.Box(
                low=np.zeros((1, lm, lm), dtype=np.float32),
                high=np.full((1, lm, lm), 4, dtype=np.float32),
            ),
            "swep_map": gym.spaces.Box(
                low=np.zeros((1, lm, lm), dtype=np.float32),
                high=np.full((1, lm, lm), 10, dtype=np.float32),
            ),
        })

        self.info = self._build_info(0, 0, 0)

    # ==================================================================
    # Gym API
    # ==================================================================
    def step(self, action):
        done = False
        self.steps += 1

        if self.state_machine == STATE_GOAL_REACHED:
            self.state_machine = STATE_WAIT_FOR_GOAL
        if self.state_machine == STATE_WAIT_FOR_GOAL:
            self.planner.set_target(self.target_list[0])
            self.target_list.pop(0)
            self.state_machine = STATE_PLANNING

        # Agent motion (no perception)
        self._update_agents()
        self.map_gt.update_dynamic_grid(self.agents)

        # Planning
        success = self.planner.plan(self.drone, self.dt)
        if not success:
            self.drone.brake()
            self.state_machine = STATE_PLANNING
            self.fail_count += 1
        else:
            self.state_machine = STATE_EXECUTING
            self.fail_count = 0

        # Control
        self.drone.step_pos(self.planner.trajectory)
        self.drone.step_yaw(action * self.config.drone_max_yaw_speed)

        # Termination
        collision = self.drone.is_colliding(self.map_gt, self.agents)
        dead_lock = 0
        freezing = 0
        if collision == 0:
            if norm(np.array([self.drone.x, self.drone.y]) - self.planner.target[:2]) <= 10:
                self.state_machine = STATE_GOAL_REACHED
            dead_lock = int(self.fail_count >= 10 and norm(self.drone.velocity) == 0)
            freezing = int(self.steps >= self.max_steps and not dead_lock)

        done = (
            collision != 0
            or bool(dead_lock)
            or bool(freezing)
            or (self.state_machine == STATE_GOAL_REACHED and len(self.target_list) == 0)
        ) or done

        if done:
            for tracker in self.drone.trackers:
                if tracker.active:
                    self.tracker_buffer.append(tracker)

        self.info = self._build_info(collision, dead_lock, freezing)
        obs = {
            "local_map": self.drone.get_local_map()[None],
            "swep_map": self.drone.get_local_map()[None],
            "yaw_angle": np.array([self.drone.yaw], dtype=np.float32),
        }
        return obs, 0, done, self.info

    def reset(self):
        self.__init__(config=self.config)
        return {}

    def render(self, mode="human"):
        if self._screen is None:
            return
        self.drone.map.render(self._screen)
        self.drone.render(self._screen)

        for ob in self.obstacles:
            pygame.draw.circle(
                self._screen, (200, 200, 200),
                center=[ob[0], ob[1]], radius=ob[2],
            )
        if len(self.planner.trajectory.positions) > 1:
            pygame.draw.lines(
                self._screen, (255, 255, 255), False,
                self.planner.trajectory.positions,
            )
        for i, agent in enumerate(self.agents):
            tracked = self.drone.trackers[i].active if i < len(self.drone.trackers) else False
            color = pygame.Color(0, 250, 250) if tracked else pygame.Color(250, 0, 0)
            if agent.group_id == 0:
                pygame.draw.circle(
                    self._screen, color,
                    np.rint(agent.position).astype(int),
                    int(round(agent.radius)),
                )
            else:
                r = int(round(agent.radius))
                pygame.draw.rect(
                    self._screen, color,
                    pygame.Rect(agent.position[0] - r, agent.position[1] - r, 2 * r, 2 * r),
                )
        for tracker in self.drone.trackers:
            if tracker.active:
                draw_covariance_ellipse(
                    self._screen,
                    tracker.mu_upds[-1][:2, 0],
                    tracker.Sigma_upds[-1][:2, :2],
                )
        pygame.draw.circle(
            self._screen, (0, 0, 255),
            self.planner.target[:2].astype(int), self.drone.radius,
        )

        font = pygame.font.SysFont("Arial", 15)
        state_names = {
            STATE_WAIT_FOR_GOAL: "WAIT_FOR_GOAL",
            STATE_GOAL_REACHED: "GOAL_REACHED",
            STATE_PLANNING: "PLANNING",
            STATE_EXECUTING: "EXECUTING",
        }
        label = font.render(
            f"STATE: {state_names.get(self.state_machine, '?')}",
            False, (0, 102, 0),
        )
        self._screen.blit(label, (0, 0))
        pygame.display.update()
        self._clock.tick(60)

    # ==================================================================
    # Metric helpers
    # ==================================================================
    def get_traversibility(self, axis_range) -> float:
        """Average traversibility over a grid of start positions."""
        from demos.traversibility import traversibility

        total = sum(
            traversibility(self.map_gt.grid_map, (x, y))
            for x, y in iter_product(axis_range, axis_range)
        )
        return total / (len(axis_range) ** 2)

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _init_obstacles(self) -> None:
        cfg = self.config

        while len(self.obstacles) < cfg.pillar_number:
            obs = np.array([
                random.randint(50, cfg.map_size[0] - 50),
                random.randint(50, cfg.map_size[1] - 50),
                random.randint(15, 20),
            ])
            ok = True
            for t in self.target_list:
                if norm(np.asarray(t) - obs[:2]) <= cfg.drone_radius + 20 + obs[2]:
                    ok = False
                    break
            if norm(np.array([self.drone.x, self.drone.y]) - obs[:2]) <= cfg.drone_radius + 70:
                ok = False
            if ok:
                self.obstacles.append(obs)

        while len(self.agents) < cfg.agent_number:
            pos = (
                random.uniform(20, cfg.map_size[0] - 20),
                random.uniform(20, cfg.map_size[1] - 20),
            )
            r = random.uniform(cfg.agent_radius - 2, cfg.agent_radius + 2)
            pref = -cfg.agent_max_speed * array([
                cos(2 * pi * len(self.agents) / max(cfg.agent_number, 1)),
                sin(2 * pi * len(self.agents) / max(cfg.agent_number, 1)),
            ])
            new_agent = Agent(pos, (0.0, 0.0), r, cfg.agent_max_speed, pref)
            ok = all(
                norm(a.position - new_agent.position) > a.radius + new_agent.radius
                for a in self.agents
            )
            ok = ok and all(
                norm(np.array(o[:2]) - new_agent.position) > o[2] + new_agent.radius + 10
                for o in self.obstacles
            )
            if norm(new_agent.position - np.array([self.drone.x, self.drone.y])) <= cfg.drone_radius + 70:
                ok = False
            if ok:
                self.agents.append(new_agent)

        try:
            shaped_map = np.load(cfg.static_map)
        except FileNotFoundError:
            return
        vels = [30 + 20 * (np.random.rand(2) - 0.5) for _ in range(100)]
        for x in range(shaped_map.shape[0]):
            for y in range(shaped_map.shape[1]):
                if shaped_map[x, y] != 0:
                    v = vels[shaped_map[x, y]]
                    self.agents.append(Agent(
                        position=np.array([5 + x * 10, 5 + y * 10]),
                        velocity=v, radius=1.414 * 5,
                        max_speed=40, pref_velocity=v,
                        group_id=shaped_map[x, y],
                    ))

    def _update_agents(self) -> None:
        cfg = self.config
        if cfg.motion_profile == "RVO":
            if self.agents and not rvo_update(self.agents, self.obstacles):
                return
            for a in self.agents:
                a.step(self.map_gt.x_scale, self.map_gt.y_scale,
                       cfg.map_size[0], cfg.map_size[1], self.dt)
        elif cfg.motion_profile == "CVM":
            for a in self.agents:
                a.velocity = a.pref_velocity
                a.step(self.map_gt.x_scale, self.map_gt.y_scale,
                       cfg.map_size[0], cfg.map_size[1], self.dt)

    def _build_info(self, collision: int, dead_lock: int, freezing: int) -> dict:
        return {
            "drone": self.drone,
            "trajectory": self.planner.trajectory,
            "state_machine": self.state_machine,
            "target": self.planner.target,
            "collision_flag": collision,
            "dead_lock_flag": dead_lock,
            "freezing_flag": freezing,
            "flight_time": self.steps * self.dt,
            "tracker_buffer": self.tracker_buffer,
        }
