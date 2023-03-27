import gym
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from numpy import array, pi, cos, sin
from numpy.linalg import norm
from traj_planner import MPC, Primitive, Jerk_Primitive
from utils import *
class Drone2DEnv2(gym.Env):
    
    @staticmethod
    def init_obstacles(self):
        while len(self.obstacles) < self.params.pillar_number:
            obs = np.array([random.randint(50,self.params.map_size[0]-50), 
                            random.randint(50,self.params.map_size[1]-50), 
                            random.randint(15,20)])
            collision_free = True
            for target in self.target_list:
                if norm(target - obs[:-1]) <= self.params.drone_radius + 20 + obs[-1]:
                    collision_free = False
                    break
            if norm(np.array([self.drone.x, self.drone.y]) - obs[:-1]) <= self.params.drone_radius + 70:
                collision_free = False
            if collision_free:
                self.obstacles.append(obs)
        
        while(len(self.agents) < self.params.agent_number):
            new_agent = Agent(position=(random.uniform(20, self.params.map_size[0]-20), random.uniform(20, self.params.map_size[1]-20)), 
                              velocity=(0., 0.), 
                              radius=self.params.agent_radius, 
                              max_speed=self.params.agent_max_speed, 
                              pref_velocity=-self.params.agent_max_speed * array([cos(2*pi*len(self.agents) / self.params.agent_number), 
                                                                             sin(2*pi*len(self.agents) / self.params.agent_number)]))
            collision_free = True
            for agent_ in self.agents:
                if norm(agent_.position - new_agent.position) <= agent_.radius + new_agent.radius:
                    collision_free = False
            for obs in self.obstacles:
                if norm(np.array([obs[0], obs[1]]) - new_agent.position) <= obs[2] + new_agent.radius + 10:
                    collision_free = False
            if collision_free:
                self.agents.append(new_agent)
     
    def __init__(self, params):
        planner_list = {
            'Primitive': Primitive,
            'MPC': MPC,
            'Jerk_Primitive':Jerk_Primitive
        }
        np.seterr(divide='ignore', invalid='ignore')
        gym.logger.set_level(40)
        plt.ion()
        random.seed(0)

        # Setup pygame environment
        if params.render:
            pygame.init()
            self.screen = pygame.display.set_mode(params.map_size)
            self.clock = pygame.time.Clock()

        self.steps = 0
        self.max_steps = params.max_steps
        self.params = params
        self.dt = params.dt
        self.tracked_agent = 0
        self.seen_history = []

        # Set target list to visit
        self.target_list = [array([params.map_size[0]/2, params.map_size[1]-(params.drone_radius+params.map_scale+20)])]

        # Generate drone
        self.drone = Drone2D(init_x=params.map_size[0]/2, 
                             init_y=params.drone_radius+params.map_scale, 
                             init_yaw=-90, 
                             dt=self.dt, 
                             params=params)

        # Generate obstacles
        self.obstacles = []
        self.agents = []
        self.init_obstacles(self)

        # Generate ground truth grid map
        self.map_gt = OccupancyGridMap(params.map_scale, params.map_size, 2)
        self.map_gt.init_obstacles(self.obstacles, self.agents)

        # Define planner
        self.planner = planner_list[params.planner](self.drone, params)
        self.state_machine = state_machine['WAIT_FOR_GOAL']
        self.fail_count = 0

        # Define action and observation space
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), shape=(1,))
        self.observation_space = gym.spaces.Dict({})
        self.info = {
            'drone':self.drone,
            'seen_agents':[agent for agent in self.agents if agent.seen],
            'trajectory':self.planner.trajectory,
            'state_machine':self.state_machine,
            'target':self.planner.target,
            'collision_flag':0
        }
        
    
    def step(self, a):
        done = False
        self.steps += 1
        # Update state machine
        if self.state_machine == state_machine['GOAL_REACHED']:
            self.state_machine = state_machine['WAIT_FOR_GOAL']
        
        # Update target point
        if self.state_machine == state_machine['WAIT_FOR_GOAL']:
            self.planner.set_target(self.target_list[-1])
            self.target_list = np.delete(arr=self.target_list, obj=-1, axis=0)
            self.state_machine = state_machine['PLANNING']
            
        # Update gridmap for dynamic obstacles
        self.map_gt.update_dynamic_grid(self.agents)

        # Raycast module
        newly_tracked, measurements = self.drone.get_measurements(self.map_gt, self.agents)
        self.tracked_agent += newly_tracked

        # Update moving agent position
        if len(self.agents) > 0:
            if RVO.RVO_update(self.agents, self.obstacles):
                for agent in self.agents:
                    agent.step(self.map_gt.x_scale, self.map_gt.y_scale, self.params.map_size[0], self.params.map_size[1],  self.dt)
            else:
                done = True
        
        # If collision detected for planned trajectory, replan
        _, swep_map = self.planner.replan_check(self.drone.map.grid_map, self.agents)

        #Plan
        success = self.planner.plan(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.drone.acceleration, self.drone.map, self.agents, self.dt)
        if not success:
            self.drone.brake()
            self.fail_count += 1
            if self.fail_count >= 6 and norm(self.drone.velocity)==0:
                done = True
        else:
            self.state_machine = state_machine['EXECUTING']
            self.fail_count = 0


        # Execute trajectory
        self.drone.step_pos(self.planner.trajectory)

        # Execute gaze control
        self.drone.step_yaw(a*self.params.drone_max_yaw_speed)

        # Check done or not
        collision_state = self.drone.is_collide(self.map_gt, self.agents)
        done = True if (collision_state != 0) or \
                       (norm(np.array([self.drone.x, self.drone.y]) - self.planner.target[:2]) <= 10 and self.target_list.shape[0] == 0) or \
                       (self.steps >= self.max_steps) else done
            
        # wrap up information
        self.seen_history.append([1 if agent.in_view else 0 for agent in self.agents])
        self.info = {
            'drone':self.drone,
            'trajectory':self.planner.trajectory,
            'swep_map':swep_map,
            'state_machine':self.state_machine,
            'collision_flag':collision_state,
            'target':self.planner.target,
            'seen_agents':[agent for agent in self.agents if agent.seen]
        }
        state = {}

        return state, 0, done, self.info
    
    def reset(self):
        self.__init__(params=self.params)
        return {}
        
    def render(self, mode='human'):
        
        # self.map_gt.render(self.screen, color_dict)
        self.drone.map.render(self.screen, color_dict)
        self.drone.render(self.screen)
        for ray in self.drone.rays:
            pygame.draw.line(
                self.screen,
                (100,100,100),
                (self.drone.x, self.drone.y),
                ((ray['coords'][0]), (ray['coords'][1]))
        )

        for ob in self.obstacles:
            pygame.draw.circle(self.screen, (200, 200, 200), center=[ob[0], ob[1]], radius=ob[2])
        
        if len(self.planner.trajectory.positions) > 1:
            pygame.draw.lines(self.screen, (100,100,100), False, self.planner.trajectory.positions)

        if len(self.agents) > 0:
            for agent in self.agents:
                agent.render(self.screen)
        pygame.draw.circle(self.screen, (0,0,255), self.planner.target[:2], self.drone.radius)
        default_font = pygame.font.SysFont('Arial', 15)
        pygame.Surface.blit(self.screen,
            default_font.render('STATE: '+list(state_machine.keys())[list(state_machine.values()).index(self.state_machine)], False, (0, 102, 0)),
            (0, 0)
        )
        
        pygame.display.update()
        self.clock.tick(60)
    
    
