from __future__ import division
import enum
from sqlalchemy import false, true
from pyorca import Agent, get_avoidance_velocity, orca, normalized, perp
from numpy import array, rint, linspace, pi, cos, sin
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

N_AGENTS = 15
RADIUS = 8.
MAX_SPEED = 4

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
        pos = (random.uniform(-20, 20), random.uniform(-20, 20))
        new_agent = Agent(pos, (0., 0.), 1., MAX_SPEED, vel)
        if check_collision(agents, new_agent):
            agents.append(new_agent)
            i += 1
    return agents


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
        
    
    def step(self):
        new_vels = [None] * len(self.agents)
        for i, agent in enumerate(self.agents):
            candidates = self.agents[:i] + self.agents[i + 1:]
            new_vels[i], _ = orca(agent, candidates, 3, self.dt)

        for i, agent in enumerate(self.agents):
            agent.velocity = new_vels[i]
            agent.position += agent.velocity * self.dt
            real_pos = agent.position * 6 + (300,200)
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
        scale = 6  # Drawing scale.
        self.screen.fill(pygame.Color(0, 0, 0))
        
        def draw_agent(agent, color):
            pygame.draw.circle(self.screen, color, rint(agent.position * scale + origin).astype(int), int(round(agent.radius * scale)), 0)
        
        def draw_velocity(a, color):
            pygame.draw.line(self.screen, color, rint(a.position * scale + origin).astype(int), rint((a.position + a.velocity) * scale + origin).astype(int), 1)
        
        for agent, color in zip(self.agents, itertools.cycle(colors)):
            draw_agent(agent, pygame.Color(0, 255, 255))

            print(norm(agent.velocity))
            
            draw_velocity(agent, pygame.Color(0, int(255 * norm(agent.velocity) / 10), 0))
        
        pygame.display.update()
        self.clock.tick(60)
    
if __name__ == '__main__':
    t = Drone2DEnv()
    while True:
        t.step()
        t.render()
        sleep(1/50)