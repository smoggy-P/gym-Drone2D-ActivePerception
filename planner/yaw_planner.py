import numpy as np
import math
from map.grid import OccupancyGridMap
from config import *
from mav.drone import Drone2D
import matplotlib.pyplot as plt

class NoControl(object):
    def __init__(self) -> None:
        pass

    def plan(self, state):
        return 0

class LookAhead(object):
    """Make the drone look at the direction of its velocity

    Args:
        object (_type_): _description_
    """
    def __init__(self, dt):
        self.dt = dt

    def plan(self, state):
        if state['drone'].velocity[1]==0 and state['drone'].velocity[0]==0:
            return 0
        
        target_yaw = math.degrees(math.atan2(-state['drone'].velocity[1], state['drone'].velocity[0])) % 360
        
        # print(target_yaw)
        if abs(target_yaw - state['drone'].yaw) < 180:
            yaw_vel = max(min((target_yaw - state['drone'].yaw) / self.dt, DRONE_MAX_YAW_SPEED), -DRONE_MAX_YAW_SPEED)
        else:
            yaw_vel = -max(min((target_yaw - state['drone'].yaw) / self.dt, DRONE_MAX_YAW_SPEED), -DRONE_MAX_YAW_SPEED)
        return yaw_vel

class Oxford(object):
    """Oxford method to plan gaze

    Args:
        object (_type_): _description_
    """
    def __init__(self, dt, dim):
        self.last_time_observed_map = np.inf * np.ones((dim[0]//MAP_GRID_SCALE, dim[1]//MAP_GRID_SCALE))
        self.swep_map = np.zeros((dim[0]//MAP_GRID_SCALE, dim[1]//MAP_GRID_SCALE))
        self.dim = dim
        self.dt = dt

        # Farthest step in trajectory that considered as priority
        self.tau_s = 3

        # Safe last time observed
        self.tau_c = 3

        self.c1 = 1000000
        self.c2 = 1000
        self.c3 = 1

        # Primitive time step
        self.v_yaw_space = np.arange(-DRONE_MAX_YAW_SPEED, DRONE_MAX_YAW_SPEED, DRONE_MAX_YAW_SPEED/3)

    def get_view_map(self, drone):
        dim = self.dim
        x = np.arange(int(dim[0]//MAP_GRID_SCALE)).reshape(-1, 1) * MAP_GRID_SCALE
        y = np.arange(int(dim[1]//MAP_GRID_SCALE)).reshape(1, -1) * MAP_GRID_SCALE

        vec_yaw = np.array([math.cos(math.radians(drone.yaw)), -math.sin(math.radians(drone.yaw))])
        view_angle = math.radians(drone.yaw_range / 2)
        #((drone.x - x)**2 + (drone.y - y)**2 <= drone.yaw_depth ** 2) and 
        # math.acos(np.array([math.cos(math.radians(drone.yaw)), -math.sin(math.radians(drone.yaw))]).dot(np.array([x - drone.x, y - drone.y]))/np.norm(np.array([x - drone.x, y - drone.y]))) <= math.radians(drone.yaw_range / 2)
        view_map = np.where(np.arccos(((x - drone.x)*vec_yaw[0] + (y - drone.y)*vec_yaw[1]) / np.sqrt((drone.x - x)**2 + (drone.y - y)**2)) <= view_angle, np.where(((drone.x - x)**2 + (drone.y - y)**2 <= drone.yaw_depth ** 2), 1, 0), 0)
        return view_map

    def plan(self, observation):
        
        drone = observation['drone']
        trajectory = observation['trajectory']
        
        # update v_i
        self.swep_map = np.zeros_like(self.swep_map)
        for i, pos in enumerate(trajectory.positions):
            self.swep_map[int(pos[0]//MAP_GRID_SCALE), int(pos[1]//MAP_GRID_SCALE)] = i * self.dt

        # update t_i
        view_map = self.get_view_map(drone)
        

        self.last_time_observed_map = np.where(view_map,
                                               0,
                                               self.last_time_observed_map + (1 - view_map) * self.dt)

        # calculate reward
        reward_map = np.where((self.swep_map > 0) & (self.swep_map <= self.tau_s) & (self.last_time_observed_map >= self.tau_c), self.c1, 
                     np.where((self.swep_map > self.tau_s) & (self.last_time_observed_map >= self.tau_c), self.c2,
                     np.clip(self.c3*self.last_time_observed_map, -np.inf, 1)))

        # calculate primitive for yaw control
        target_yaw = drone.yaw + self.v_yaw_space * self.dt
        max_reward = 0
        best_action = 0

        if len(trajectory)==0:
            return 0

        for i, yaw in enumerate(target_yaw):
            new_drone = Drone2D(trajectory.positions[0][0], trajectory.positions[0][1], yaw, self.dt, self.dim)
            new_view_map = self.get_view_map(new_drone)
            if max_reward < np.sum(new_view_map*reward_map):
                best_action = i
                max_reward = np.sum(new_view_map*reward_map)
        
        return self.v_yaw_space[best_action]


        # plt.imshow(reward_map)
        # plt.show()
        # plt.pause(0.1)
        # plt.clf()

        
        
        


        

        


    