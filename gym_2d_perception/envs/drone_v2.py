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
     
    def __init__(self, params):
        planner_list = {
            'Primitive': Primitive,
            'MPC': MPC,
            'Jerk_Primitive':Jerk_Primitive
        }
        np.seterr(divide='ignore', invalid='ignore')
        gym.logger.set_level(40)
        plt.ion()

        self.steps = 0
        self.max_steps = params.max_steps
        self.params = params
        self.dt = params.dt
        self.tracked_agent = 0
        self.seen_history = []

        # Setup pygame environment
        if params.render:
            pygame.init()
            self.screen = pygame.display.set_mode(params.map_size)
            self.clock = pygame.time.Clock()
        
        # Set target list to visit, random order
        self.target_list = np.array([[params.map_size[0]/2, params.map_size[1]-(params.drone_radius+params.map_scale+20)]])

        # Generate drone
        self.drone = Drone2D(init_x=params.map_size[0]/2, 
                             init_y=params.drone_radius+params.map_scale, 
                             init_yaw=-90, 
                             dt=self.dt, 
                             params=params)

        # Generate pillars and make sure it do not overlap with start and target point
        self.obstacles = []
        while len(self.obstacles) < params.pillar_number:
            obs = np.array([random.randint(50,params.map_size[0]-50), 
                            random.randint(50,params.map_size[1]-50), 
                            random.randint(15,20)])
            collision_free = True
            for target in self.target_list:
                if norm(target - obs[:-1]) <= params.drone_radius + 20 + obs[-1]:
                    collision_free = False
                    break
            if norm(np.array([self.drone.x, self.drone.y]) - obs[:-1]) <= params.drone_radius + 70:
                collision_free = False
            if collision_free:
                self.obstacles.append(obs)
        
        # Generate dynamic obstacles
        self.agents = []
        while(len(self.agents) < params.agent_number):
            x = array([cos(2*pi*len(self.agents) / params.agent_number), 
                       sin(2*pi*len(self.agents) / params.agent_number)])
            vel = -x * params.agent_max_speed
            pos = (random.uniform(20, params.map_size[0]-20), random.uniform(20, params.map_size[1]-20))
            new_agent = Agent(position=pos, 
                              velocity=(0., 0.), 
                              radius=params.agent_radius, 
                              max_speed=params.agent_max_speed, 
                              pref_velocity=vel)
            if check_collision(self.agents, new_agent, self.obstacles):
                self.agents.append(new_agent)

        # Generate ground truth grid map
        self.map_gt = OccupancyGridMap(params.map_scale, params.map_size, 2)
        self.map_gt.init_obstacles(self.obstacles, self.agents)

        # Define planner
        self.planner = planner_list[params.planner](self.drone, params)
        self.swep_map = np.zeros(array(params.map_size)//params.map_scale)
        self.state_machine = state_machine['WAIT_FOR_GOAL']
        self.fail_count = 0

        # Define action and observation space
        self.info = {
            'drone':self.drone,
            'seen_agents':[agent for agent in self.agents if agent.seen],
            'trajectory':self.planner.trajectory,
            'state_machine':self.state_machine,
            'target':self.planner.target,
            'collision_flag':0
        }
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), shape=(1,))
        local_map_size = 4 * (params.drone_view_depth // params.map_scale) + 1
        self.observation_space = gym.spaces.Dict(
            {
                'yaw_angle' : gym.spaces.Box(low=np.array([0], dtype=np.float32), 
                                             high=np.array([360], dtype=np.float32), 
                                             shape=(1,), 
                                             dtype=np.float32), 
                'local_map' : gym.spaces.Box(low=np.zeros((1, local_map_size, local_map_size), dtype=np.float32), 
                                             high=np.float32(4*np.ones((1, local_map_size,local_map_size))), 
                                             shape=(1, local_map_size, local_map_size),
                                             dtype=np.float32),
                'swep_map'  : gym.spaces.Box(low=np.zeros((1, local_map_size, local_map_size), dtype=np.float32), 
                                             high=np.float32(10*np.ones((1, local_map_size,local_map_size), dtype=np.float32)), 
                                             shape=(1, local_map_size, local_map_size),
                                             dtype=np.float32)
            }
        )
    
    def step(self, a):
        done = False
        self.steps += 1
        # Update state machine
        if self.state_machine == state_machine['GOAL_REACHED']:
            self.state_machine = state_machine['WAIT_FOR_GOAL']
        # Update gridmap for dynamic obstacles
        self.map_gt.update_dynamic_grid(self.agents)

        # Raycast module
        newly_tracked = self.drone.raycasting(self.map_gt, self.agents)
        self.tracked_agent += newly_tracked

        # Update moving agent position
        if len(self.agents) > 0:
            if RVO.RVO_update(self.agents, self.obstacles):
                for agent in self.agents:
                    agent.step(self.map_gt.x_scale, self.map_gt.y_scale, self.params.map_size[0], self.params.map_size[1],  self.dt)
            else:
                done = True
        
        # If collision detected for planned trajectory, replan
        replan, swep_map = self.planner.replan_check(self.drone.map.grid_map, self.agents)

        # Set target point
        if self.state_machine == state_machine['WAIT_FOR_GOAL']:
            self.planner.set_target(self.target_list[-1])
            self.target_list = np.delete(arr=self.target_list, obj=-1, axis=0)
            self.state_machine = state_machine['PLANNING']

        #Plan
        # if self.state_machine == state_machine['PLANNING']:
        success = self.planner.plan(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.drone.acceleration, self.drone.map, self.agents, self.dt)
        if not success:
            self.drone.brake()
            self.fail_count += 1
            # print("fail plan, fail count:", self.fail_count)
            if self.fail_count >= 3 and norm(self.drone.velocity)==0:
                done = True
        else:
            self.state_changed = True
            self.state_machine = state_machine['EXECUTING']
            self.fail_count = 0


        # Execute trajectory
        if self.planner.trajectory.positions != [] :
            self.drone.acceleration = self.planner.trajectory.accelerations[0]
            self.drone.velocity = self.planner.trajectory.velocities[0]
            self.drone.x = round(self.planner.trajectory.positions[0][0])
            self.drone.y = round(self.planner.trajectory.positions[0][1])
            self.planner.trajectory.pop()
            if norm(np.array([self.drone.x, self.drone.y]) - self.planner.target[:2]) <= 10:
                self.planner.trajectory.positions = []
                self.planner.trajectory.velocities = []
                self.state_machine = state_machine['GOAL_REACHED']
        # Execute gaze control
        self.drone.step_yaw(a*self.params.drone_max_yaw_speed)

        # Return reward
        collision_state = self.drone.is_collide(self.map_gt, self.agents)
        if collision_state == 1:
            if self.params.record_img and self.params.gaze_method != 'NoControl':
                pygame.image.save(self.screen, self.params.img_dir+self.params.gaze_method+'_static_'+ str(datetime.now())+'.png')
            done = True
        elif collision_state == 2:
            if self.params.record_img and self.params.gaze_method != 'NoControl':
                pygame.image.save(self.screen, self.params.img_dir+self.params.gaze_method+'_dynamic_'+ str(datetime.now())+'.png')
            done = True
        elif self.state_machine == state_machine['GOAL_REACHED']:
            done = False
            if self.target_list.shape[0] == 0:
                done = True
        if self.steps >= self.max_steps:
             done = True
            
        # wrap up information
        self.seen_history.append([1 if agent.in_view else 0 for agent in self.agents])
        self.info = {
            'drone':self.drone,
            'trajectory':self.planner.trajectory,
            'swep_map':self.swep_map,
            'state_machine':self.state_machine,
            'collision_flag':collision_state,
            'target':self.planner.target,
            'seen_agents':[agent for agent in self.agents if agent.seen]
        }
        drone_idx = (int(self.drone.x // self.params.map_scale), int(self.drone.y // self.params.map_scale))
        edge_len = 2 * (self.params.drone_view_depth // self.params.map_scale)
        local_swep_map = np.pad(swep_map, ((edge_len,edge_len),(edge_len,edge_len)), 'constant', constant_values=0)
        local_swep_map = local_swep_map[drone_idx[0] : drone_idx[0] + 2 * edge_len + 1, drone_idx[1] : drone_idx[1] + 2 * edge_len + 1]

        state = {
            'local_map' : self.drone.get_local_map()[None],
            'swep_map' : local_swep_map[None],
            'yaw_angle' : np.array([self.drone.yaw], dtype=np.float32).flatten()
        }

        return state, 0, done, self.info
    
    def reset(self):
        self.__init__(params=self.params)
        local_map_size = 4 * (self.params.drone_view_depth // self.params.map_scale) + 1
        return {'local_map' : self.drone.get_local_map()[None], 
                'swep_map'  : np.zeros((1, local_map_size, local_map_size)),
                'yaw_angle' : np.array([self.drone.yaw])}
        
    def render(self, mode='human'):
        
        # self.map_gt.render(self.screen, color_dict)
        self.drone.map.render(self.screen, color_dict)
        self.drone.render(self.screen)
        # for ray in self.drone.rays:
        #     pygame.draw.line(
        #         self.screen,
        #         (100,100,100),
        #         (self.drone.x, self.drone.y),
        #         ((ray['coords'][0]), (ray['coords'][1]))
        # )
        draw_static_obstacle(self.screen, self.obstacles, (200, 200, 200))
        
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
    
    
