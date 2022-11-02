import math
import gym
import pygame
import random

from numpy import array, pi, cos, sin
from RVO import RVO_update, Agent
from grid import OccupancyGridMap
from raycast import Raycast
from drone import Drone2D
from time import sleep
from primitive import Primitive, Trajectory2D


from utils import *
from config import *      

class Drone2DEnv(gym.Env):
     
    def __init__(self):
        
        self.dt = 1/15
        
        self.obstacles = {
            'circular_obstacles'  : [[320, 240, 50]],
            'rectangle_obstacles' : [[100, 100, 100, 40], [400, 300, 100, 30]]
        }
        
        # Define workspace model for RVO model (approximate using circles)
        self.ws_model = obs_dict_to_ws_model(self.obstacles)
        
        # Setup pygame environment
        pygame.init()
        self.dim = MAP_SIZE
        self.screen = pygame.display.set_mode(self.dim)
        self.clock = pygame.time.Clock()
        
        # Define action and observation space
        self.action_space = None
        self.observation_space = None
        
        # Define physical setup
        self.agents = []
        if ENABLE_DYNAMIC:
            i = 1
            while(i <= N_AGENTS):
                theta = 2 * pi * i / N_AGENTS
                x = array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
                vel = -x * PEDESTRIAN_MAX_SPEED
                pos = (random.uniform(200, 440), random.uniform(120, 360))
                new_agent = Agent(pos, (0., 0.), PEDESTRIAN_RADIUS, PEDESTRIAN_MAX_SPEED, vel, self.dt)
                if check_collision(self.agents, new_agent, self.ws_model):
                    self.agents.append(new_agent)
                    i += 1

        self.map_gt = OccupancyGridMap(64, 48, self.dim)
        self.map_gt.init_obstacles(self.obstacles, self.agents)
            
        self.drone = Drone2D(self.dim[0] / 2, DRONE_RADIUS + self.map_gt.x_scale, 270, self.dt)
        self.raycast = Raycast(self.dim, self.drone)  
        self.planner = Primitive(self.screen)
        self.trajectory = Trajectory2D()
        self.planning = False
    
    def step(self):
        # Update gridmap for dynamic obstacles
        self.map_gt.update_dynamic_grid(self.agents)

        # Raycast module
        self.rays = self.raycast.castRays(self.drone, self.map_gt, self.drone.map)
        
        # Set target point
        mouse = pygame.mouse.get_pressed()
        if mouse[0]:
            success = False
            x, y = pygame.mouse.get_pos()
            self.planner.set_target(np.array([x, y]))
            print("target set as:", x, y)
            self.trajectory, success = self.planner.planning(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.map_gt, self.agents, self.dt)
            if not success:
                self.drone.brake()
                print("path not found, replanning")
                self.planning = True
            else:
                print("path found")
                self.planning = False
        
        if self.planning:
            self.trajectory, success = self.planner.planning(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.map_gt, self.agents, self.dt)
            if not success:
                self.drone.brake()
                print("path not found, replanning")
            else:
                print("path found")
                self.planning = False

        # execute trajectory
        if self.trajectory.positions != [] :
            self.drone.velocity = self.trajectory.velocities[0]
            self.drone.x = round(self.trajectory.positions[0][0])
            self.drone.y = round(self.trajectory.positions[0][1])
            self.trajectory.pop()


        if ENABLE_DYNAMIC:
            # Update moving agent position
            RVO_update(self.agents, self.ws_model)
            for agent in self.agents:
                agent.step(self.map_gt.x_scale, self.map_gt.y_scale, self.dim[0], self.dim[1], self.drone, self.dt)
        
        reward = 10
        
        done = False
        
        return reward, done, {}
    
    def reset(self):
        return self.state
        
    def render(self, mode='human'):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.drone.yaw += 2
        if keys[pygame.K_RIGHT]:
            self.drone.yaw -= 2
        pygame.event.pump() # process event queue
        
        # self.map_gt.render(self.screen, color_dict)
        self.drone.map.render(self.screen, color_dict)
        self.drone.render(self.screen)
        for ray in self.rays:
            pygame.draw.line(
                self.screen,
                (100,100,100),
                (self.drone.x, self.drone.y),
                ((ray['coords'][0]), (ray['coords'][1]))
        )
        draw_static_obstacle(self.screen, self.obstacles, (200, 200, 200))
        
        if len(self.trajectory.positions) > 1:
            pygame.draw.lines(self.screen, (100,100,100), False, self.trajectory.positions)
        # for point in self.waypoints:
        #     pygame.draw.circle(self.screen, (0, 0, 255), point, 3)

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
        # sleep(t.dt)