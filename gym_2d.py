from __future__ import division
import math
from pyorca import Agent, orca, normalized
from numpy import array, rint, pi, cos, sin
from numpy.linalg import norm
from time import sleep
import gym
import pygame
import random
import numpy as np


N_AGENTS = 15
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
        pos = (random.uniform(200, 440), random.uniform(120, 360))
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
        self.drone = Drone2D(320, 240, 15)
        
    
    def step(self):
        new_vels = [None] * len(self.agents)
        for i, agent in enumerate(self.agents):
            candidates = self.agents[:i] + self.agents[i + 1:]
            new_vels[i], _ = orca(agent, candidates, 1, self.dt)

        for i, agent in enumerate(self.agents):
            agent.velocity = new_vels[i]
            agent.position += agent.velocity * self.dt
            
            # Turn around if out of boundary
            if agent.position[0] < 0 or agent.position[0] > 640:
                agent.velocity[0] = -agent.velocity[0]
            if agent.position[1] < 0 or agent.position[1] > 480:
                agent.velocity[1] = -agent.velocity[1]
                
            # Check if the pedestrian is seen
            vec_yaw = np.array([cos(math.radians(self.drone.yaw)), -sin(math.radians(self.drone.yaw))])
            vec_agent = np.array([agent.position[0] - self.drone.x, agent.position[1] - self.drone.y])
            if norm(agent.position - (self.drone.x, self.drone.y)) <= self.drone.yaw_depth and math.acos(vec_yaw.dot(vec_agent)/norm(vec_agent)) <= math.radians(self.drone.yaw_range / 2):
                agent.seen = True
            else:
                agent.seen = False
        reward = 10
        
        done = False
        
        return self.agents, reward, done, {}
    
    def reset(self):
        self.agents = init_agents()
        return self.state
        
    def render(self, mode='human'):
        scale = 1  # Drawing scale.
        self.screen.fill(pygame.Color(0, 0, 0))
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.drone.yaw += 2
        if keys[pygame.K_RIGHT]:
            self.drone.yaw -= 2
        if keys[pygame.K_UP]:
            self.drone.x += 5*cos(math.radians(self.drone.yaw))
            self.drone.y -= 5*sin(math.radians(self.drone.yaw))
        pygame.event.pump() # process event queue
        
        def draw_agent(agent, color):
            pygame.draw.circle(self.screen, color, rint(agent.position * scale).astype(int), int(round(agent.radius * scale)), 0)
        
        def draw_velocity(a, color):
            pygame.draw.line(self.screen, color, rint(a.position * scale).astype(int), rint((a.position + a.velocity) * scale).astype(int), 1)
        
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
            
        
        for agent in self.agents:
            if agent.seen:
                draw_agent(agent, pygame.Color(0, 255, 255))
            else:
                draw_agent(agent, pygame.Color(255, 0, 0))
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