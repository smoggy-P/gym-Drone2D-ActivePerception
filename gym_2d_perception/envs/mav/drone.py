import sys
sys.path.append('/home/smoggy/thesis/gym-Drone2D-ActivePerception/gym_2d_perception/envs')

from map.grid import OccupancyGridMap
from map.utils import *
from config import *
from mav.raycast import Raycast

from numpy.linalg import norm
import numpy as np

class Drone2D():
    def __init__(self, init_x, init_y, init_yaw, dt, dim):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.yaw_range = 90
        self.yaw_depth = 80
        self.radius = DRONE_RADIUS
        self.map = OccupancyGridMap(MAP_GRID_SCALE, dim, 0)
        self.view_map = np.zeros((dim[0]//MAP_GRID_SCALE, dim[1]//MAP_GRID_SCALE))
        self.velocity = np.array([0, 0])
        self.dt = dt
        self.rays = {}
        self.raycast = Raycast(dim, self)

    def step_yaw(self, action):
        self.yaw = (self.yaw + action * self.dt) % 360

    def raycasting(self, gt_map, agents):
        self.view_map = np.zeros_like(self.view_map)
        self.rays = self.raycast.castRays(self, gt_map, agents)

    def brake(self):
        if norm(self.velocity) <= DRONE_MAX_ACC * self.dt:
            self.velocity = np.zeros(2)

        else:
            self.velocity = self.velocity - self.velocity / norm(self.velocity) * DRONE_MAX_ACC * self.dt
            self.x += self.velocity[0] * self.dt
            self.y += self.velocity[1] * self.dt

    def is_collide(self, gt_map, agents):
        grid = gt_map.get_grid(self.x, self.y)
        if grid == grid_type['OCCUPIED']:
            # print("collision with static obstacles")
            return True
        for agent in agents:
            if norm(agent.position - np.array([self.x, self.y])) < agent.radius + self.radius:
                # print("collision with dynamic obstacles")
                return True

        return False

    def render(self, surface):
        pygame.draw.arc(surface, 
                        (255,255,255), 
                        [self.x - self.yaw_depth,
                            self.y - self.yaw_depth,
                            2 * self.yaw_depth,
                            2 * self.yaw_depth], 
                        math.radians(self.yaw - self.yaw_range/2), 
                        math.radians(self.yaw + self.yaw_range/2),
                        2)
        angle1 = math.radians(self.yaw + self.yaw_range/2)
        angle2 = math.radians(self.yaw - self.yaw_range/2)
        pygame.draw.line(surface, (255,255,255), (self.x, self.y), (self.x + self.yaw_depth * cos(angle1), self.y - self.yaw_depth * sin(angle1)), 2)
        pygame.draw.line(surface, (255,255,255), (self.x, self.y), (self.x + self.yaw_depth * cos(angle2), self.y - self.yaw_depth * sin(angle2)), 2)
        pygame.draw.circle(surface, (255,255,255), (self.x, self.y), self.radius)