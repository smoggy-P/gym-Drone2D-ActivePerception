import math
import gym
import pygame
import random
import numpy as np

from numpy import array, rint, pi, cos, sin
from numpy.linalg import norm
from time import sleep
from RVO import RVO_update, reach, compute_V_des, reach, Agent
from grid import OccupancyGridMap


N_AGENTS = 3
RADIUS = 8.
MAX_SPEED = 30

def check_collision(agents, agent, ws_model):
    for agent_ in agents:
        if norm(agent_.position - agent.position) <= agent_.radius + agent.radius:
            return False
    for obs in ws_model['circular_obstacles']:
        if norm(np.array([obs[0], obs[1]]) - agent.position) <= obs[2] + agent.radius:
            return False
    return True

def check_in_view(drone, position):
    # Check if the target point is seen
    vec_yaw = np.array([cos(math.radians(drone.yaw)), -sin(math.radians(drone.yaw))])
    vec_agent = np.array([position[0] - drone.x, position[1] - drone.y])
    if norm(position - (drone.x, drone.y)) <= drone.yaw_depth and math.acos(vec_yaw.dot(vec_agent)/norm(vec_agent)) <= math.radians(drone.yaw_range / 2):
        return True
    else:
        return False
    

def init_agents(ws_model):
    agents = []
    i = 1
    while(i <= N_AGENTS):
        theta = 2 * pi * i / N_AGENTS
        x = array((cos(theta), sin(theta))) #+ random.uniform(-1, 1)
        vel = -x * MAX_SPEED
        pos = (random.uniform(200, 440), random.uniform(120, 360))
        new_agent = Agent(pos, (0., 0.), 6., MAX_SPEED, vel)
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
        self.yaw_depth = 100
        
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
        

class Drone2DEnv(gym.Env):
     
    def __init__(self):
        
        #define workspace model
        ws_model = dict()
        #circular obstacles, format [x,y,rad]
        # no obstacles
        # ws_model['circular_obstacles'] = [[50 + 10*i, 240, 5] for i in range(20)] + [[400, i*10, 5] for i in range(20)]
        
        ws_model['circular_obstacles'] = [[320, 240, 50]]
        
        # with obstacles
        # ws_model['circular_obstacles'] = [[-0.3, 2.5, 0.3], [1.5, 2.5, 0.3], [3.3, 2.5, 0.3], [5.1, 2.5, 0.3]]
        #rectangular boundary, format [x,y,width/2,heigth/2]
        ws_model['boundary'] = [] 
        
        self.ws_model = ws_model
        
        self.dt = 1/20
        self.agents = init_agents(self.ws_model)
        
        self.action_space = None
        self.observation_space = None
        
        self.dim = (640, 480)
        pygame.init()
        self.screen = pygame.display.set_mode(self.dim)
        self.clock = pygame.time.Clock()
        self.drone = Drone2D(320, 0, 270)
        self.global_map = OccupancyGridMap(64, 48, self.dim)
    
    def step(self):

        RVO_update(self.agents, self.ws_model)
        
        for i, agent in enumerate(self.agents):
            new_position = agent.position + np.array(agent.velocity) * self.dt
            
            # Change reference velocity if reaching the boundary
            if new_position[0] < 0 or new_position[0] > self.dim[0]:
                agent.velocity[0] = -agent.velocity[0]
            if new_position[1] < 0 or new_position[1] > self.dim[1]:
                agent.velocity[1] = -agent.velocity[1]
                
            agent.position += np.array(agent.velocity) * self.dt
            
            # Check if the pedestrian is seen
            if check_in_view(self.drone, agent.position):
                agent.seen = True
            else:
                agent.seen = False
        
        # Update grid map
        for i in range(self.global_map.width):
            for j in range(self.global_map.height):
                if(check_in_view(self.drone, self.global_map.real_pos(i, j))):
                    self.global_map.grid_map[i, j] = 2
        
        reward = 10
        
        done = False
        
        return self.agents, reward, done, {}
    
    def reset(self):
        self.agents = init_agents()
        return self.state
        
    def render(self, mode='human'):
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
        
        def draw_static_obstacle(ws_model, color):
            for obs in ws_model['circular_obstacles']:
                pygame.draw.circle(self.screen, color, center=[obs[0], obs[1]], radius=obs[2])
        
        def draw_agent(agent, color):
            pygame.draw.circle(self.screen, color, rint(agent.position).astype(int), int(round(agent.radius)), 0)
        
        def draw_velocity(a, color):
            pygame.draw.line(self.screen, color, rint(a.position).astype(int), rint((a.position + a.velocity)).astype(int), 1)
            
        color_dict = {
            'OCCUPIED'   : (150, 150, 150),
            'UNOCCUPIED' : (50, 50, 50),
            'UNEXPLORED' : (0, 0, 0)
        }
        
        self.global_map.render(self.screen, color_dict)
        self.drone.render(self.screen)
        
        for agent in self.agents:
            if agent.seen:
                draw_agent(agent, pygame.Color(0, 255, 255))
            else:
                draw_agent(agent, pygame.Color(255, 0, 0))
            draw_velocity(agent, pygame.Color(0, 255, 0))
        
        draw_static_obstacle(self.ws_model, (200, 200, 200))
        
        pygame.display.update()
        self.clock.tick(60)
    
if __name__ == '__main__':
    t = Drone2DEnv()
    while True:
        t.step()
        t.render()