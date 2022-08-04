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
        x = RADIUS * array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
        vel = normalized(-x) * MAX_SPEED
        pos = (random.uniform(-20, 20), random.uniform(-20, 20))
        new_agent = Agent(pos, (0., 0.), 1., MAX_SPEED, vel)
        if check_collision(agents, new_agent):
            agents.append(new_agent)
            i += 1
    return agents


class Drone2DEnv(gym.Env):
    
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 10
    }
     
    def __init__(self):
        self.dt = 1/20
        self.agents = init_agents()
        self.action_space = None
        self.observation_space = None
        self.viewer = None
        
    
    def step(self):
        new_vels = [None] * len(self.agents)
        for i, agent in enumerate(self.agents):
            candidates = self.agents[:i] + self.agents[i + 1:]
            new_vels[i], _ = orca(agent, candidates, 1, self.dt)

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
        
        if self.viewer is None:
            self.viewer = rendering.Viewer(600, 400)
            self.pedtrans = []
            circle = []
            for i, agent in enumerate(self.agents):
                circle.append(rendering.make_circle(8))
                self.pedtrans.append(rendering.Transform())
                circle[i].add_attr(self.pedtrans[i])
                self.viewer.add_geom(circle[i])
        for i, agent in enumerate(self.agents):
            self.pedtrans[i].set_translation(agent.position[0] * 6 + 300, 
                                             agent.position[1] * 6 + 200)
        
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
if __name__ == '__main__':
    t = Drone2DEnv()
    while True:
        t.step()
        t.render()
        sleep(1/50)