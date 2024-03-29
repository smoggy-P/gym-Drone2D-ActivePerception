import gym
import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, pi, cos, sin
from numpy.linalg import norm
from traj_planner import MPC, Primitive, Jerk_Primitive, NoMove
from utils import *
class Drone2DEnv2(gym.Env):
    
    @staticmethod
    def init_obstacles_random_size(self):
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
                              radius=random.uniform(5, 15) if self.params.agent_radius == -1 else random.uniform(self.params.agent_radius-2, self.params.agent_radius+2), 
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
            if norm(new_agent.position - np.array([self.drone.x, self.drone.y])) <= self.params.drone_radius + 70:
                collision_free = False

            if collision_free:
                self.drone.trackers[len(self.agents)].radius = new_agent.radius
                self.agents.append(new_agent)
        
        shaped_obs_map = np.load(self.params.static_map)

        vels = []
        for i in range(100):
            vel = self.params.agent_max_speed
            direction = np.random.rand() * 2 * np.pi
            vels.append([vel * np.cos(direction), vel * np.sin(direction)])
        
        for x in range(shaped_obs_map.shape[0]):
            for y in range(shaped_obs_map.shape[1]):
                if shaped_obs_map[x][y] != 0:
                    vel = vels[shaped_obs_map[x][y]]
                    self.agents.append(Agent(position=np.array([5 + x * 10,5 + y * 10]), 
                                            velocity=vel, 
                                            radius=5, 
                                            max_speed=vel, 
                                            pref_velocity=vel,
                                            group_id=shaped_obs_map[x][y]))

     
    def __init__(self, params):
        planner_list = {
            'Primitive': Primitive,
            'MPC': MPC,
            'Jerk_Primitive':Jerk_Primitive,
            'NoMove':NoMove
        }
        np.seterr(divide='ignore', invalid='ignore')
        gym.logger.set_level(40)
        plt.ion()
        random.seed(params.map_id)
        np.random.seed(params.map_id)

        # Setup pygame environment
        if params.render:
            pygame.init()
            self.screen = pygame.display.set_mode(params.map_size)
            self.clock = pygame.time.Clock()

        self.steps = 0
        self.max_steps = params.max_flight_time / params.dt
        self.params = params
        self.dt = params.dt
        self.tracked_agent = 0
        self.tracker_buffer = []

        # Set target list to visit
        self.target_list = params.target_list.copy()

        # Generate drone
        self.drone = Drone2D(init_x=params.init_position[0], 
                             init_y=params.init_position[1], 
                             init_yaw=-90, 
                             dt=self.dt, 
                             params=params)

        # Generate obstacles
        self.obstacles = []
        self.agents = []
        self.init_obstacles_random_size(self)

        # Generate ground truth grid map
        self.map_gt = OccupancyGridMap(params.map_scale, params.map_size, 2)
        self.map_gt.init_obstacles(self.obstacles, self.agents)

        # Define planner
        self.planner = planner_list[params.planner](self.drone, params)
        self.state_machine = state_machine['WAIT_FOR_GOAL']
        self.fail_count = 0

        # Define action and observation space
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), shape=(1,))
        
        self.info = {
            'drone':self.drone,
            'trajectory':self.planner.trajectory,
            'state_machine':self.state_machine,
            'target':self.planner.target,
            'collision_flag':0,
            'dead_lock_flag':0,
            'freezing_flag':0,
            'flight_time':self.steps * self.dt,
            'tracker_buffer':self.tracker_buffer
        }
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
        
        # Update target point
        if self.state_machine == state_machine['WAIT_FOR_GOAL']:
            self.planner.set_target(self.target_list[0])
            self.target_list.pop(0)
            self.state_machine = state_machine['PLANNING']

        ##########################
        ### Environment module ###
        ##########################
        # Update moving agent position
        if self.params.motion_profile == "RVO":
            if len(self.agents) > 0:
                if RVO.RVO_update(self.agents, self.obstacles):
                    for agent in self.agents:
                        agent.step(self.map_gt.x_scale, self.map_gt.y_scale, self.params.map_size[0], self.params.map_size[1],  self.dt)
                else:
                    done = True
        elif self.params.motion_profile == "CVM":
            for agent in self.agents:
                agent.velocity = agent.pref_velocity
                agent.step(self.map_gt.x_scale, self.map_gt.y_scale, self.params.map_size[0], self.params.map_size[1],  self.dt)

        #########################
        ### Perception module ###
        #########################
        newly_tracked, measurements = self.drone.get_measurements(self.map_gt, self.agents)
        # Update gridmap for dynamic obstacles
        self.map_gt.update_dynamic_grid(self.agents)
        self.tracker_buffer.extend(self.drone.update_tracker(measurements))
        self.tracked_agent += newly_tracked
        
        #######################
        ### Planning module ###
        #######################
        # If collision detected for planned trajectory, replan
        _, swep_map = self.planner.replan_check(self.drone)

        #Plan
        success = self.planner.plan(self.drone, self.dt)
        if not success:
            self.drone.brake()
            self.state_machine = state_machine['PLANNING']
            self.fail_count += 1
        else:
            self.state_machine = state_machine['EXECUTING']
            self.fail_count = 0


        ######################
        ### Control module ###
        ######################
        # Execute trajectory
        self.drone.step_pos(self.planner.trajectory)

        # Execute gaze control
        self.drone.step_yaw(a*self.params.drone_max_yaw_speed)

        # Check done or not
        collision_state = self.drone.is_collide(self.map_gt, self.agents)

        dead_lock_flag = 0
        freezing_flag = 0

        if collision_state == 0:
            self.state_machine = state_machine['GOAL_REACHED'] if norm(np.array([self.drone.x, self.drone.y]) - self.planner.target[:2]) <= 10 else self.state_machine
            dead_lock_flag = 1 if (self.fail_count >= 10) and (norm(self.drone.velocity)==0) else 0 
            freezing_flag = 1 if (self.steps >= self.max_steps) and (not dead_lock_flag) else 0


        done = True if (collision_state != 0) or \
                        dead_lock_flag or \
                        freezing_flag or \
                        ((self.state_machine == state_machine['GOAL_REACHED']) and len(self.target_list) == 0) else done
        if done:
            for tracker in self.drone.trackers:
                if tracker.active == True:
                    self.tracker_buffer.append(tracker)
            
        # wrap up information
        self.info = {
            'drone':self.drone,
            'trajectory':self.planner.trajectory,
            'state_machine':self.state_machine,
            'target':self.planner.target,

            'collision_flag':collision_state,
            'dead_lock_flag':dead_lock_flag,
            'freezing_flag':freezing_flag,
            
            'flight_time':self.steps * self.dt,
            'tracker_buffer':self.tracker_buffer
        }
        state = {
            'local_map' : self.drone.get_local_map()[None],
            'swep_map' : self.drone.get_local_map()[None],
            'yaw_angle' : np.array([self.drone.yaw], dtype=np.float32).flatten()
        }

        return state, 0, done, self.info
    
    def reset(self):
        self.__init__(params=self.params)
        return {}
        
    def render(self, mode='human'):
        
        self.drone.map.render(self.screen, color_dict)
        # self.map_gt.render(self.screen, color_dict)
        self.drone.render(self.screen)

        for ob in self.obstacles:
            pygame.draw.circle(self.screen, (130, 130, 130), center=[ob[0], ob[1]], radius=ob[2])
        
        if len(self.planner.trajectory.positions) > 1:
            pygame.draw.lines(self.screen, (0,0,0), False, self.planner.trajectory.positions, 2)
        if hasattr(self.planner, 'future_trajectory') and len(self.planner.future_trajectory.positions) > 1:
            pygame.draw.lines(self.screen, (0,0,0), False, self.planner.future_trajectory.positions, 2)

        if len(self.agents) > 0:
            for i, agent in enumerate(self.agents):
                color = pygame.Color(130, 176, 210) if self.drone.trackers[i].active else pygame.Color(200, 36, 35)
                if agent.group_id >= 0:
                    pygame.draw.circle(self.screen, color, np.rint(agent.position).astype(int), int(round(agent.radius)), 0)
                    pygame.draw.lines(self.screen, (53, 53, 53), False, [np.rint(agent.position).astype(int), np.rint(agent.position + agent.velocity).astype(int)], 2)
                else:
                    pygame.draw.rect(self.screen, color, pygame.Rect(agent.position[0]-agent.radius, agent.position[1]-agent.radius, 2*agent.radius, 2*agent.radius))
        
        # for tracker in self.drone.trackers:
        #     if tracker.active:
        #         pygame.draw.circle(self.screen, (100, 20, 20), center=[tracker.mu_upds[-1][0,0], tracker.mu_upds[-1][1,0]], radius=4)
        #         draw_cov(self.screen, tracker.mu_upds[-1][:2,0], tracker.Sigma_upds[-1][:2,:2])

        # Drawing the star as target
        star_points = calculate_star_points(self.planner.target[:2], self.drone.radius, self.drone.radius * 0.5)
        pygame.draw.polygon(self.screen, (0, 0, 150), star_points)


        # default_font = pygame.font.SysFont('Arial', 15)
        # pygame.Surface.blit(self.screen,
        #     default_font.render('STATE: ' + list(state_machine.keys())[list(state_machine.values()).index(self.state_machine)], False, (0, 80, 0)), # Darker Green for text
        #     (0, 0)
        # )
        
        pygame.display.update()
        self.clock.tick(60)
    
    
