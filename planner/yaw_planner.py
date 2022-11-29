import numpy as np
import math
from map.grid import OccupancyGridMap
from config import *
import matplotlib.pyplot as plt

class LookAhead(object):
    """Make the drone look at the direction of its velocity

    Args:
        object (_type_): _description_
    """
    def __init__(self) -> None:
        pass

    def plan(self, drone, trajectory):
        drone.yaw = math.degrees(math.atan2(-drone.velocity[1], drone.velocity[0]))

class Oxford(object):
    """Oxford method to plan gaze

    Args:
        object (_type_): _description_
    """
    def __init__(self, dt, dim):
        self.last_time_observed_map = np.inf * np.ones((dim[0]//MAP_GRID_SCALE, dim[1]//MAP_GRID_SCALE))
        self.swep_map = np.zeros((dim[0]//MAP_GRID_SCALE, dim[1]//MAP_GRID_SCALE))
        self.dt = dt

        # Farthest step in trajectory that considered as priority
        self.tau_s = 3

        # Safe last time observed
        self.tau_c = 3

        self.c1 = 5
        self.c2 = 2
        self.c3 = 1

        # Primitive time step
        self.t_ptimitive = 0.5
        self.v_yaw_space = np.arange(-DRONE_MAX_YAW_SPEED, DRONE_MAX_YAW_SPEED, DRONE_MAX_YAW_SPEED/3)

    def plan(self, drone, trajectory):
        # update v_i
        self.swep_map = np.zeros_like(self.swep_map)
        for i, pos in enumerate(trajectory.positions):
            self.swep_map[int(pos[0]//MAP_GRID_SCALE), int(pos[1]//MAP_GRID_SCALE)] = i * self.dt

        # update t_i
        x_drone = int(drone.x // MAP_GRID_SCALE)
        y_drone = int(drone.y // MAP_GRID_SCALE)
        x, y = np.ogrid[:self.swep_map.shape[0], :self.swep_map.shape[1]]

        self.last_time_observed_map = np.where(drone.view_map,
                                               0,
                                               self.last_time_observed_map + (1 - drone.view_map) * self.dt)

        # calculate reward
        reward_map = np.where((self.swep_map > 0) & (self.swep_map <= self.tau_s) & (self.last_time_observed_map >= self.tau_c), self.c1, 
                     np.where((self.swep_map > self.tau_s) & (self.last_time_observed_map >= self.tau_c), self.c2,
                     np.clip(self.c3*self.last_time_observed_map, -np.inf, 1)))
        
        plt.imshow(reward_map)
        plt.show()
        plt.pause(0.1)
        plt.clf()

        # calculate primitive for yaw control
        # T = len(trajectory.position) * self.dt
        


        

        


    