import numpy as np
import math
import matplotlib.pyplot as plt
import random
from gym_2d_perception.envs.drone_v0 import Drone2D

class NoControl(object):
    def __init__(self, params):
        self.params = params

    def plan(self, state):
        # self.v_yaw_space = np.arange(-self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed/3)
        return 0

class LookAhead(object):
    """Make the drone look at the direction of its velocity

    Args:
        object (_type_): _description_
    """
    def __init__(self, params):
        self.dt = params.dt
        self.params = params

    def plan(self, state):
        if state['drone'].velocity[1]==0 and state['drone'].velocity[0]==0:
            return 0
        
        target_yaw = math.degrees(math.atan2(-state['drone'].velocity[1], state['drone'].velocity[0])) % 360
        
        # print(target_yaw)
        if abs(target_yaw - state['drone'].yaw) < 180:
            yaw_vel = max(min((target_yaw - state['drone'].yaw) / self.dt, self.params.drone_max_yaw_speed), -self.params.drone_max_yaw_speed)
        else:
            yaw_vel = -max(min((target_yaw - state['drone'].yaw) / self.dt, self.params.drone_max_yaw_speed), -self.params.drone_max_yaw_speed)
        return yaw_vel / self.params.drone_max_yaw_speed

class Oxford(object):
    """Oxford method to plan gaze

    Args:
        object (_type_): _description_
    """
    def __init__(self, params):
        self.params = params
        self.last_time_observed_map = 5 * np.ones((params.map_size[0]//params.map_scale, params.map_size[1]//params.map_scale))
        self.swep_map = np.zeros((params.map_size[0]//params.map_scale, params.map_size[1]//params.map_scale))
        self.dim = params.map_size
        self.dt = params.dt

        # Farthest step in trajectory that considered as priority
        self.tau_s = 3

        # Safe last time observed
        self.tau_c = 0.5

        self.c1 = 1000000
        self.c2 = 1000
        self.c3 = 1

        # Primitive time step
        self.v_yaw_space = np.arange(-self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed/3)

    def get_view_map(self, drone):
        dim = self.dim
        x = np.arange(int(dim[0]//self.params.map_scale)).reshape(-1, 1) * self.params.map_scale
        y = np.arange(int(dim[1]//self.params.map_scale)).reshape(1, -1) * self.params.map_scale

        vec_yaw = np.array([math.cos(math.radians(drone.yaw)), -math.sin(math.radians(drone.yaw))])
        view_angle = math.radians(drone.yaw_range / 2)
        #((drone.x - x)**2 + (drone.y - y)**2 <= drone.yaw_depth ** 2) and 
        
        # math.acos(np.array([math.cos(math.radians(drone.yaw)), -math.sin(math.radians(drone.yaw))]).dot(np.array([x - drone.x, y - drone.y]))/np.norm(np.array([x - drone.x, y - drone.y]))) <= math.radians(drone.yaw_range / 2)
        np.seterr(divide='ignore', invalid='ignore')
        view_map = np.where(np.logical_or((drone.x - x)**2 + (drone.y - y)**2 <= 0, np.logical_and(np.arccos(((x - drone.x)*vec_yaw[0] + (y - drone.y)*vec_yaw[1]) / np.sqrt((drone.x - x)**2 + (drone.y - y)**2)) <= view_angle, ((drone.x - x)**2 + (drone.y - y)**2 <= drone.yaw_depth ** 2))), 1, 0)
        return view_map

    def plan(self, observation):
        
        drone = observation['drone']
        trajectory = observation['trajectory']
        
        # update v_i
        self.swep_map = np.zeros_like(self.swep_map)
        for i, pos in enumerate(trajectory.positions):
            self.swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.dt

        # update t_i
        view_map = self.get_view_map(self, drone)
        

        self.last_time_observed_map = np.where(view_map,
                                               0,
                                               self.last_time_observed_map + (1 - view_map) * self.dt)

        plt.subplot(1,2,1)
        plt.imshow(self.last_time_observed_map.T)
        plt.subplot(1,2,2)
        plt.imshow(self.swep_map.T)
        plt.show()
        plt.pause(0.001)
        plt.clf()

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
            new_drone = Drone2D(trajectory.positions[0][0], trajectory.positions[0][1], yaw, self.dt, self.dim, self.params)
            new_view_map = self.get_view_map(self, new_drone)
            if max_reward < np.sum(new_view_map*reward_map):
                best_action = i
                max_reward = np.sum(new_view_map*reward_map)
        
        return self.v_yaw_space[best_action] / self.params.drone_max_yaw_speed


        # plt.imshow(reward_map)
        # plt.show()
        # plt.pause(0.1)
        # plt.clf()

        
class Rotating(object):
    
    def __init__(self, params):
        self.params = params
    
    def plan(self, observation):
        return 1


        

        


    