import math
from turtle import Screen
import gym
import pygame
import random
import numpy as np

from numpy import array, pi, cos, sin
from RVO import RVO_update, Agent
from grid import OccupancyGridMap
from raycast import Raycast
from utils import *
from config import *

def init_agents(ws_model):
    agents = []
    i = 1
    while(i <= N_AGENTS):
        theta = 2 * pi * i / N_AGENTS
        x = array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
        vel = -x * PEDESTRIAN_MAX_SPEED
        pos = (random.uniform(200, 440), random.uniform(120, 360))
        new_agent = Agent(pos, (0., 0.), PEDESTRIAN_RADIUS, PEDESTRIAN_MAX_SPEED, vel)
        if check_collision(agents, new_agent, ws_model):
            agents.append(new_agent)
            i += 1
    return agents



class Drone2D():
    def __init__(self, init_x, init_y, init_yaw):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.yaw_range = 120
        self.yaw_depth = 150
        self.radius = DRONE_RADIUS
        
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
        

class Drone2DEnv(gym.Env):
     
    def __init__(self):
        
        self.dt = 1/20
        
        self.obstacles = {
            'circular_obstacles'  : [[320, 240, 50]],
            'rectangle_obstacles' : [[100, 100, 100, 40], [400, 300, 100, 30]]
        }
        
        # Define workspace model for RVO model (approximate using circles)
        self.ws_model = obs_dict_to_ws_model(self.obstacles)
        
        # Setup pygame environment
        pygame.init()
        self.dim = (640, 480)
        self.screen = pygame.display.set_mode(self.dim)
        self.clock = pygame.time.Clock()
        
        # Define action and observation space
        self.action_space = None
        self.observation_space = None
        
        # Define physical setup
        self.global_map = OccupancyGridMap(64, 48, self.dim, self.obstacles)
        
        if ENABLE_DYNAMIC:
            self.agents = init_agents(self.ws_model)
            
        self.drone = Drone2D(self.dim[0] / 2, DRONE_RADIUS + self.global_map.x_scale, 270)
        self.raycast = Raycast(self.dim, self.drone)
        
    
    def step(self):
        self.rays = self.raycast.castRays(self.drone, self.global_map)
        if ENABLE_DYNAMIC:
            # Update moving agent position
            RVO_update(self.agents, self.ws_model)
            for agent in self.agents:
                agent.step(self.global_map.x_scale, self.global_map.y_scale, self.dim[0], self.dim[1], self.drone, self.dt)
        
        reward = 10
        
        done = False
        
        return reward, done, {}
    
    def reset(self):
        if ENABLE_DYNAMIC:
            self.agents = init_agents()
        return self.state
        
    def render(self, mode='human'):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.drone.yaw += 2
        if keys[pygame.K_RIGHT]:
            self.drone.yaw -= 2
        if keys[pygame.K_UP]:
            self.drone.x += 5*cos(math.radians(self.drone.yaw))
            self.drone.y -= 5*sin(math.radians(self.drone.yaw))
        pygame.event.pump() # process event queue
        
        self.global_map.render(self.screen, color_dict)
        self.drone.render(self.screen)
        for ray in self.rays:
            pygame.draw.line(
                self.screen,
                (100,100,100),
                (self.drone.x, self.drone.y),
                ((ray['coords'][0]), (ray['coords'][1]))
        )
        draw_static_obstacle(self.screen, self.obstacles, (200, 200, 200))
        
        
        if ENABLE_DYNAMIC:
            for agent in self.agents:
                agent.render(self.screen)
        
        fps = round(self.clock.get_fps())
        if (fps >= 40):
            fps_color = (0,102,0)
        elif(fps >= 20):
            fps_color = (255, 153, 0)
        else:
            fps_color = (204, 0, 0)
        default_font = pygame.font.SysFont('Arial', 15)
        pygame.Surface.blit(self.screen,
            default_font.render('FPS: '+str(fps), False, fps_color),
            (0, 0)
        )
        
        pygame.display.update()
        self.clock.tick(60)
    
if __name__ == '__main__':
    t = Drone2DEnv()
    while True:
        t.step()
        t.render()