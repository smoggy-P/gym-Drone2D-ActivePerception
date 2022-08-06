from __future__ import division
import enum
import math
from sqlalchemy import false, true
from pyorca import Agent, get_avoidance_velocity, orca, normalized, perp
from numpy import angle, array, rint, linspace, pi, cos, sin
from numpy.linalg import norm
from gym.envs.classic_control import rendering
from time import sleep
import gym
import pygame
import itertools
import random

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
]

N_AGENTS = 10
RADIUS = 8.
MAX_SPEED = 24

def check_collision(agents, agent):
    for agent_ in agents:
        if norm(agent_.position - agent.position) <= agent_.radius + agent.radius:
            return False
    return True
    

def init_agents():
    agents = []
    i = 0
    while(i <= N_AGENTS):
        theta = 2 * pi * i / N_AGENTS
        x = array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
        vel = normalized(-x) * MAX_SPEED
        pos = (random.uniform(-120, 120), random.uniform(-120, 120))
        new_agent = Agent(pos, (0., 0.), 6., MAX_SPEED, vel)
        if check_collision(agents, new_agent):
            agents.append(new_agent)
            i += 1
    return agents

class Drone2D():
    def __init__(self, init_x, init_y, init_yaw):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.yaw_range = 120
        self.yaw_depth = 100

class Drone2DEnv(gym.Env):
     
    def __init__(self):
        self.dt = 1/20
        self.agents = init_agents()
        
        self.action_space = None
        self.observation_space = None
        
        self.dim = (640, 480)
        pygame.init()
        self.screen = pygame.display.set_mode(self.dim)
        self.clock = pygame.time.Clock()
        self.drone = Drone2D(320, 240, 0)
        
    
    def step(self):
        new_vels = [None] * len(self.agents)
        for i, agent in enumerate(self.agents):
            candidates = self.agents[:i] + self.agents[i + 1:]
            new_vels[i], _ = orca(agent, candidates, 1, self.dt)

        for i, agent in enumerate(self.agents):
            agent.velocity = new_vels[i]
            agent.position += agent.velocity * self.dt
            real_pos = agent.position + (300,200)
            if real_pos[0] < 0 or real_pos[0] > 600:
                agent.velocity[0] = -agent.velocity[0]
            if real_pos[1] < 0 or real_pos[1] > 400:
                agent.velocity[1] = -agent.velocity[1]
        
        reward = 10
        
        done = False
        
        return self.agents, reward, done, {}
    
    def reset(self):
        self.agents = init_agents()
        return self.state
        
    def render(self, mode='human'):
        
        origin = array(self.dim) / 2  # Screen position of origin.
        scale = 1  # Drawing scale.
        self.screen.fill(pygame.Color(0, 0, 0))
        
        def draw_agent(agent, color):
            pygame.draw.circle(self.screen, color, rint(agent.position * scale + origin).astype(int), int(round(agent.radius * scale)), 0)
        
        def draw_velocity(a, color):
            pygame.draw.line(self.screen, color, rint(a.position * scale + origin).astype(int), rint((a.position + a.velocity) * scale + origin).astype(int), 1)
        
        def draw_drone(drone):
            pygame.draw.arc(self.screen, 
                            (255,255,255), 
                            [drone.x - drone.yaw_depth,
                             drone.y - drone.yaw_depth,
                             2 * drone.yaw_depth,
                             2 * drone.yaw_depth], 
                            math.radians(drone.yaw - drone.yaw_range/2), 
                            math.radians(drone.yaw + drone.yaw_range/2),
                            2)
            angle1 = math.radians(drone.yaw + drone.yaw_range/2)
            angle2 = math.radians(drone.yaw - drone.yaw_range/2)
            pygame.draw.line(self.screen, (255,255,255), (self.drone.x, self.drone.y), (self.drone.x + self.drone.yaw_depth * cos(angle1), self.drone.y - self.drone.yaw_depth * sin(angle1)), 2)
            pygame.draw.line(self.screen, (255,255,255), (self.drone.x, self.drone.y), (self.drone.x + self.drone.yaw_depth * cos(angle2), self.drone.y - self.drone.yaw_depth * sin(angle2)), 2)
            
        
        for agent, color in zip(self.agents, itertools.cycle(colors)):
            draw_agent(agent, pygame.Color(0, 255, 255))
            
            draw_velocity(agent, pygame.Color(0, 255, 0))
        
        draw_drone(self.drone)
        pygame.display.update()
        self.clock.tick(60)
    
if __name__ == '__main__':
    t = Drone2DEnv()
    while True:
        t.step()
        t.render()
        sleep(1/50)