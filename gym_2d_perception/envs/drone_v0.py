import sys
sys.path.append('/home/smoggy/thesis/gym-Drone2D-ActivePerception/gym_2d_perception/envs')

import gym
import pygame
import random
import matplotlib.pyplot as plt
import pylab as pl

from numpy import array, pi, cos, sin
from map.RVO import RVO_update, Agent
from map.grid import OccupancyGridMap
from mav.drone import Drone2D
from planner.primitive import Primitive, Trajectory2D
# from planner.yaw_planner import LookAhead, Oxford, NoControl
from IPython import display


from map.utils import *    
color_dict = {
    'OCCUPIED'   : (150, 150, 150),
    'UNOCCUPIED' : (50, 50, 50),
    'UNEXPLORED' : (0, 0, 0)
}

state_machine = {
        'WAIT_FOR_GOAL':0,
        'GOAL_REACHED' :1,
        'PLANNING'     :2,
        'EXECUTING'    :3
    }
class Drone2DEnv(gym.Env):
     
    def __init__(self, params):
        self.params = params
        self.dt = params.dt
        
        self.obstacles = {
            'circular_obstacles'  : [[320, 240, 50]],
            'rectangle_obstacles' : [[100, 100, 100, 40], [400, 300, 100, 30]]
        }
        
        # Define workspace model for RVO model (approximate using circles)
        self.ws_model = obs_dict_to_ws_model(self.obstacles)
        
        # Setup pygame environment
        self.dim = params.map_size
        self.is_render = params.render
        if self.is_render:
            pygame.init()
            self.screen = pygame.display.set_mode(self.dim)
            self.clock = pygame.time.Clock()
        
        # Define physical setup
        self.agents = []
        i = 1
        while(i <= params.agent_number):
            theta = 2 * pi * i / params.agent_number
            x = array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
            vel = -x * params.agent_max_speed
            pos = (random.uniform(self.dim[0] / 2 - 100, self.dim[0] / 2 +100), random.uniform(self.dim[1] / 2 - 100, self.dim[1] / 2 +100))
            new_agent = Agent(pos, (0., 0.), params.agent_radius, params.agent_max_speed, vel, self.dt)
            if check_collision(self.agents, new_agent, self.ws_model):
                self.agents.append(new_agent)
                i += 1

        self.map_gt = OccupancyGridMap(params.map_scale, self.dim, 2)
        self.map_gt.init_obstacles(self.obstacles, self.agents)
    
        self.drone = Drone2D(self.dim[0] / 2, params.drone_radius + self.map_gt.x_scale, -90, self.dt, self.dim, params)
        self.planner = Primitive(self.drone, params)

        self.target_list = [np.array([520, 100]), np.array([120, 50]), np.array([120, 380]), np.array([520, 380])]
        
        self.trajectory = Trajectory2D()
        self.state = state_machine['WAIT_FOR_GOAL']
        self.state_changed = False
        self.replan_count = 0

        # Define action and observation space
        self.observation = {
            'drone':self.drone,
            'trajectory':self.trajectory
        }
    
    def step(self, a):
        done = False
        self.state_changed = False
        # Update state machine
        if self.state == state_machine['GOAL_REACHED']:
            self.state = state_machine['WAIT_FOR_GOAL']
            self.state_changed = True
        # Update gridmap for dynamic obstacles
        self.map_gt.update_dynamic_grid(self.agents)

        # Raycast module
        self.drone.raycasting(self.map_gt, self.agents)

        # Update moving agent position
        if len(self.agents) > 0:
            if RVO_update(self.agents, self.ws_model):
                for agent in self.agents:
                    agent.step(self.map_gt.x_scale, self.map_gt.y_scale, self.dim[0], self.dim[1],  self.dt)
            else:
                done = True
        
        # Set target point
        if self.state == state_machine['WAIT_FOR_GOAL']:
            self.planner.set_target(self.target_list[-1])
            self.target_list.pop()
            self.state = state_machine['PLANNING']
        # mouse = pygame.mouse.get_pressed()
        # if mouse[0]:
        #     success = False
        #     x, y = pygame.mouse.get_pos()
        #     self.planner.set_target(np.array([x, y]))
        #     self.trajectory, success = self.planner.plan(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.drone.map, self.agents, self.dt)
        #     self.state_changed = True
        #     if not success:
        #         self.drone.brake()
        #         self.state = state_machine['PLANNING']
        #     else:
        #         self.state = state_machine['EXECUTING']

        #Plan
        if self.state == state_machine['PLANNING']:
            self.trajectory, success = self.planner.plan(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.drone.map, self.agents, self.dt)
            if not success:
                self.drone.brake()
                # print("path not found, replanning")
            else:
                # print("path found")
                self.state_changed = True
                self.state = state_machine['EXECUTING']

        # If collision detected for planned trajectory, replan
        swep_map = np.zeros_like(self.map_gt.grid_map)
        for i, pos in enumerate(self.trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.dt
            for agent in self.agents:
                if agent.seen:
                    estimate_pos = agent.estimate_pos + i * self.dt * agent.estimate_vel
                    if norm(estimate_pos - pos) <= self.drone.radius + agent.radius:
                        self.state = state_machine['PLANNING']
                        self.state_changed = True   
        obs_map = np.where((self.drone.map.grid_map==0) | (self.drone.map.grid_map==2), 0, 1)
        if np.sum(obs_map * swep_map) > 0:
            self.state = state_machine['PLANNING']
            self.state_changed = True

        # Execute trajectory
        if self.trajectory.positions != [] :
            self.drone.velocity = self.trajectory.velocities[0]
            self.drone.x = round(self.trajectory.positions[0][0])
            self.drone.y = round(self.trajectory.positions[0][1])
            self.trajectory.pop()
            if self.trajectory.positions == []:
                self.state_changed = True
                self.state = state_machine['GOAL_REACHED']
        
        # Execute gaze control
        self.drone.step_yaw(a)
        
        # Print state machine
        # if self.state_changed:
        #     if self.state == state_machine['GOAL_REACHED']:
        #         print("state: goal reached")
        #     elif self.state == state_machine['WAIT_FOR_GOAL']:
        #         print("state: wait for goal")
        #     elif self.state == state_machine['PLANNING']:
        #         print("state: planning")
        #     elif self.state == state_machine['EXECUTING']:
        #         print("state: executing trajectory")

        # wrap up observation
        self.observation = {
            'drone':self.drone,
            'trajectory':self.trajectory
        }

        # Return reward
        if self.drone.is_collide(self.map_gt, self.agents):
            reward = -100
            done = True
        elif self.state == state_machine['GOAL_REACHED']:
            reward = 100
            done = False
            if len(self.target_list) == 0:
                done = True
        else:
            reward = -1

        return self.observation, reward, done
    
    def reset(self):
        self.__init__(params=self.params)
        
    def render(self, mode='human'):
        # keys = pygame.key.get_pressed()
        # if keys[pygame.K_LEFT]:
        #     self.drone.yaw += 2
        # if keys[pygame.K_RIGHT]:
        #     self.drone.yaw -= 2
        # pygame.event.pump() # process event queue
        
        # self.map_gt.render(self.screen, color_dict)
        if self.is_render:
            self.drone.map.render(self.screen, color_dict)
            self.drone.render(self.screen)
            for ray in self.drone.rays:
                pygame.draw.line(
                    self.screen,
                    (100,100,100),
                    (self.drone.x, self.drone.y),
                    ((ray['coords'][0]), (ray['coords'][1]))
            )
            # draw_static_obstacle(self.screen, self.obstacles, (200, 200, 200))
            
            if len(self.trajectory.positions) > 1:
                pygame.draw.lines(self.screen, (100,100,100), False, self.trajectory.positions)

            if len(self.agents) > 0:
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
    
# if __name__ == '__main__':
#     env = Drone2DEnv(render=True)
#     # policy = LookAhead(env.dt)
#     policy = Oxford(env.dt, env.dim)
#     # plt.ion()
#     max_step = 10000
#     rewards = []
#     steps = []
#     for i in range(max_step):
        
#         a = policy.plan(env.observation)
#         observation, reward, done= env.step(a)

#         if done:
#             env.reset()

#         env.render()
        # sleep(t.dt)
    # plt.plot(rewards)
    # plt.show()
    