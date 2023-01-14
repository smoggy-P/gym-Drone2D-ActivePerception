import gym
import pygame
import random
import torch
import math
import numpy as np
import time
from datetime import datetime
from numpy import array, pi, cos, sin
from numpy.linalg import norm
from math import cos, sin, atan2, asin, sqrt, radians, tan, ceil, atan, degrees
from heapq import heappush, heappop
color_dict = {
    'OCCUPIED'   : (150, 150, 150),
    'UNOCCUPIED' : (50, 50, 50),
    'UNEXPLORED' : (0, 0, 0)
}

state_machine = {
        'WAIT_FOR_GOAL':0,
        'GOAL_REACHED' :1,
        'PLANNING'     :2,
        'EXECUTING'    :3,
        'STATIC_COLLISION'     :4,
        'DYNAMIC_COLLISION'    :5,
    }

grid_type = {
    'DYNAMIC_OCCUPIED' : 3,
    'OCCUPIED' : 1,
    'UNOCCUPIED' : 2,
    'UNEXPLORED' : 0
}

def waypoint_from_traj(coeff, t):
    """Get the waypoint in trajectory with coefficient at time t

    Args:
        coeff (_type_): _description_
        t (_type_): _description_

    Returns:
        _type_: _description_
    """
    waypoint = Waypoint2D()
    waypoint.position = np.around(np.array([1, t, t**2]) @ coeff.T)
    waypoint.velocity = np.array([1, 2*t]) @ coeff[:, 1:].T
    return waypoint

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

def obs_dict_to_ws_model(obs_dict):
    """Transfer obstacle dictionary to ws_model; approximate the rectangle

    Args:
        obs_dict (dictionary): obstacles dictionary with keys of "rectangle_obstacles" and "circular_obstacles"
    """
    
    ws_model = {
        'circular_obstacles' : []
    }
    
    ws_model['circular_obstacles'] += obs_dict['circular_obstacles']
    
    for rect in obs_dict['rectangle_obstacles']:
        edge1, edge2 = rect[2:]
        edge_max = max(edge1, edge2)
        edge_min = min(edge1, edge2)
        
        if edge_max <= 2 *edge_min:
            ws_model['circular_obstacles'].append([rect[0]+rect[2]/2, rect[1]+rect[3]/2, edge_max/2])
        elif edge1 > edge2:
            ws_model['circular_obstacles'].append([rect[0]+rect[2]/4, rect[1]+rect[3]/2, edge_max/4])
            ws_model['circular_obstacles'].append([rect[0]+rect[2]/4*3, rect[1]+rect[3]/2, edge_max/4])
        else:
            ws_model['circular_obstacles'].append([rect[0]+rect[2]/2, rect[1]+rect[3]/4, edge_max/4])
            ws_model['circular_obstacles'].append([rect[0]+rect[2]/2, rect[1]+rect[3]/4*3, edge_max/4])
    
    return ws_model
    
def draw_static_obstacle(surface, obs_dict, color):
    if 'circular_obstacles' in obs_dict:
        for obs in obs_dict['circular_obstacles']:
            pygame.draw.circle(surface, color, center=[obs[0], obs[1]], radius=obs[2])
    if 'rectangle_obstacles' in obs_dict:
        for obs in obs_dict['rectangle_obstacles']:
            pygame.draw.rect(surface, color, obs)

def RVO_update(agents, ws_model):
    """ compute best velocity given the desired velocity, current velocity and workspace model"""
    
    X = [agent.position for agent in agents]
    V_des = [agent.pref_velocity for agent in agents]
    V_current = [agent.velocity for agent in agents]
    
    ROB_RAD = agents[0].radius+0.01
       
    for i in range(len(X)):
        try:
            RVO_BA_all = [[[X[i][0]+0.5*(V_current[j][0]+V_current[i][0]), X[i][1]+0.5*(V_current[j][1]+V_current[i][1])], 
                        [cos(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) + asin(2*ROB_RAD/distance(X[i], X[j]))), sin(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) + asin(2*ROB_RAD/distance(X[i], X[j])))],
                        [cos(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) - asin(2*ROB_RAD/distance(X[i], X[j]))), sin(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) - asin(2*ROB_RAD/distance(X[i], X[j])))],
                        distance(X[i], X[j]),
                        2.2*ROB_RAD] for j in range(len(X)) if j != i] + [[X[i], 
                                                                            [cos(atan2(hole[1]-X[i][1], hole[0]-X[i][0]) + asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2]))), sin(atan2(hole[1]-X[i][1], hole[0]-X[i][0]) + asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2])))], 
                                                                            [cos(atan2(hole[1]-X[i][1], hole[0]-X[i][0])-asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2]))), sin(atan2(hole[1]-X[i][1], hole[0]-X[i][0])-asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2])))], 
                                                                            distance(X[i], hole[0:2]), 
                                                                            hole[2]*1.5+ROB_RAD] for hole in ws_model['circular_obstacles']]
        except:
            return False
        vA_post = intersect(X[i], V_des[i], RVO_BA_all)
        agents[i].velocity = np.array(vA_post[:])
    return True

def distance(pose1, pose2):
    """ compute Euclidean distance for 2D """
    return sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)+0.001

def intersect(pA, vA, RVO_BA_all):
    # print '----------------------------------------'
    # print 'Start intersection test'
    norm_v = distance(vA, [0, 0])
    suitable_V = []
    unsuitable_V = []
    for theta in np.arange(0, 2*3.14, 0.2):
        for rad in np.arange(0.02, norm_v+0.02, norm_v/5.0):
            new_v = [rad*cos(theta), rad*sin(theta)]
            suit = True
            for RVO_BA in RVO_BA_all:
                theta_dif = atan2(new_v[1]+pA[1]-RVO_BA[0][1], new_v[0]+pA[0]-RVO_BA[0][0])
                theta_right = atan2(RVO_BA[2][1], RVO_BA[2][0])
                theta_left = atan2(RVO_BA[1][1], RVO_BA[1][0])
                if in_between(theta_right, theta_dif, theta_left):
                    suit = False
                    break
            if suit:
                suitable_V.append(new_v)
            else:
                unsuitable_V.append(new_v)                
    new_v = vA[:]
    suit = True
    for RVO_BA in RVO_BA_all:
        theta_dif = atan2(new_v[1]+pA[1]-RVO_BA[0][1], new_v[0]+pA[0]-RVO_BA[0][0])
        theta_right = atan2(RVO_BA[2][1], RVO_BA[2][0])
        theta_left = atan2(RVO_BA[1][1], RVO_BA[1][0])
        if in_between(theta_right, theta_dif, theta_left):
            suit = False
            break
    if suit:
        suitable_V.append(new_v)
    else:
        unsuitable_V.append(new_v)
    #----------------------        
    if suitable_V:
        # print 'Suitable found'
        vA_post = min(suitable_V, key = lambda v: distance(v, vA))
        new_v = vA_post[:]
        for RVO_BA in RVO_BA_all:
            theta_dif = atan2(new_v[1]+pA[1]-RVO_BA[0][1], new_v[0]+pA[0]-RVO_BA[0][0])
            theta_right = atan2(RVO_BA[2][1], RVO_BA[2][0])
            theta_left = atan2(RVO_BA[1][1], RVO_BA[1][0])
    else:
        # print 'Suitable not found'
        tc_V = dict()
        for unsuit_v in unsuitable_V:
            tc_V[tuple(unsuit_v)] = 0
            tc = []
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dist = RVO_BA[3]
                rad = RVO_BA[4]
                dif = [unsuit_v[0]+pA[0]-p_0[0], unsuit_v[1]+pA[1]-p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
                if in_between(theta_right, theta_dif, theta_left):
                    small_theta = abs(theta_dif-0.5*(theta_left+theta_right))
                    if abs(dist*sin(small_theta)) >= rad:
                        rad = abs(dist*sin(small_theta))
                    big_theta = asin(abs(dist*sin(small_theta))/rad)
                    dist_tg = abs(dist*cos(small_theta))-abs(rad*cos(big_theta))
                    if dist_tg < 0:
                        dist_tg = 0                    
                    tc_v = dist_tg/distance(dif, [0,0])
                    tc.append(tc_v)
            tc_V[tuple(unsuit_v)] = min(tc)+0.001
        WT = 0.2
        vA_post = min(unsuitable_V, key = lambda v: ((WT/tc_V[tuple(v)])+distance(v, vA)))
    return vA_post 

def in_between(theta_right, theta_dif, theta_left):
    if abs(theta_right - theta_left) <= 3.14:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left <0) and (theta_right >0):
            theta_left += 2*3.14
            if theta_dif < 0:
                theta_dif += 2*3.14
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left >0) and (theta_right <0):
            theta_right += 2*3.14
            if theta_dif < 0:
                theta_dif += 2*3.14
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False
            

        return False
class Agent(object):
    """A disk-shaped agent."""
    def __init__(self, position, velocity, radius, max_speed, pref_velocity, dt):
        super(Agent, self).__init__()
        self.dt = dt
        self.position = np.array(position, dtype=np.float)
        self.velocity = np.array(velocity, dtype=np.float)
        self.radius = radius
        self.max_speed = max_speed
        self.pref_velocity = np.array(pref_velocity)
        self.seen = False
        self.in_view = False
        self.var = 0
        self.estimate_vel = np.array([0,0])
        self.estimate_pos = np.array([0,0])
        
    def step(self, edge_size_x, edge_size_y, map_width, map_height, dt):
        new_position = self.position + self.velocity * dt
            
        # Change reference velocity if reaching the boundary
        if new_position[0] < edge_size_x + self.radius:
            self.pref_velocity[0] = abs(self.pref_velocity[0])
            self.seen = False
        elif new_position[0] > map_width - edge_size_x - self.radius:
            self.pref_velocity[0] = -abs(self.pref_velocity[0])
            self.seen = False
        if new_position[1] < edge_size_y + self.radius:
            self.pref_velocity[1] = abs(self.pref_velocity[1])
            self.seen = False
        elif new_position[1] > map_height - edge_size_y - self.radius:
            self.pref_velocity[1] = -abs(self.pref_velocity[1])
            self.seen = False
            
        self.position += np.array(self.velocity) * dt
        
        # Check if the pedestrian is seen
        if self.in_view is True:
            self.var = 0
            self.estimate_vel = self.velocity
            self.estimate_pos = self.position
        else:
            if self.seen:
                self.var += self.dt*10
                self.estimate_pos = self.estimate_pos + self.estimate_vel * dt

            
        
    def render(self, surface):
        if self.seen:
            pygame.draw.circle(surface, pygame.Color(0, 250, 250), np.rint(self.position).astype(int), int(round(self.radius)), 0)
            pygame.draw.circle(surface, pygame.Color(250, 0, 0), np.rint(self.estimate_pos).astype(int), int(round(self.radius+self.var)), 1)
            pygame.draw.line(surface, pygame.Color(250, 0, 0), np.rint(self.position).astype(int), np.rint(self.estimate_pos).astype(int), 1)
        else:
            pygame.draw.circle(surface, pygame.Color(250, 0, 0), np.rint(self.position).astype(int), int(round(self.radius)), 0)
        pygame.draw.line(surface, pygame.Color(0, 255, 0), np.rint(self.position).astype(int), np.rint((self.position + self.velocity)).astype(int), 1)
class OccupancyGridMap:
    def __init__(self, grid_scale, dim, init_num):
        self.dim = dim
        self.width = dim[0] // grid_scale
        self.height = dim[1] // grid_scale
        
        self.x_scale = grid_scale
        self.y_scale = grid_scale
        
        # Define Grid Map
        self.grid_map = np.ones((self.width, self.height), dtype=np.uint8) * init_num

        self.dynamic_idx = []
        
    def init_obstacles(self, obstacles_dict, agents):
        # Mark edges in Grid Map
        self.grid_map[0,:] = grid_type['OCCUPIED']
        self.grid_map[-1,:] = grid_type['OCCUPIED']
        self.grid_map[:,0] = grid_type['OCCUPIED']
        self.grid_map[:,-1] = grid_type['OCCUPIED']
        
        # Mark static obstacles in Grid Map
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                if 'circular_obstacles' in obstacles_dict:
                    for circle in obstacles_dict['circular_obstacles']:
                        if norm(self.get_real_pos(i,j) - np.array([circle[0], circle[1]])) <= circle[2]:
                            self.grid_map[i,j] = grid_type['OCCUPIED']
                if 'rectangle_obstacles' in obstacles_dict:
                    for rect in obstacles_dict['rectangle_obstacles']:
                        if rect[0] <= self.get_real_pos(i,j)[0] <= rect[0] + rect[2] and rect[1] <= self.get_real_pos(i,j)[1] <= rect[1] + rect[3]:
                            self.grid_map[i,j] = grid_type['OCCUPIED']
                if len(agents) > 0:
                    for agent in agents:
                        if (self.get_real_pos(i,j)[0] - agent.position[0])**2 + (self.get_real_pos(i,j)[1] - agent.position[1])**2 <= agent.radius ** 2:
                            self.grid_map[i,j] = grid_type['DYNAMIC_OCCUPIED']
                            self.dynamic_idx.append([i,j])
    
    def update_dynamic_grid(self, agents):
        for dynamic_idx in self.dynamic_idx:
            self.grid_map[dynamic_idx[0], dynamic_idx[1]] = grid_type['UNEXPLORED']
        self.dynamic_idx = []
        
        for agent in agents:
            unit_x = int(agent.radius // self.x_scale)
            unit_y = int(agent.radius // self.y_scale)
            pos = [int(agent.position[0] // self.x_scale), int(agent.position[1] // self.y_scale)]
            for i in range(max(pos[0] - unit_x, 0), min(pos[0] + unit_x + 1, self.grid_map.shape[0])):
                for j in range(max(pos[1] - unit_y, 0), min(pos[1] + unit_y + 1, self.grid_map.shape[1])):
                    if self.grid_map[i,j] != grid_type['OCCUPIED']:
                        self.grid_map[i,j] = grid_type['DYNAMIC_OCCUPIED']
                        self.dynamic_idx.append([i,j])

    
    def get_real_pos(self, i, j):
        return np.array([self.x_scale * (i+0.5), self.y_scale * (j+0.5)])
    
    def get_grid(self, x, y):
        if x >= self.dim[0] or x < 0 or y >= self.dim[1] or y < 0:
            return 1
        return self.grid_map[int(x // self.x_scale), int(y // self.y_scale)]
    
    
    def render(self, surface, color_dict):
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                if(self.grid_map[i,j] == grid_type['OCCUPIED'] or self.grid_map[i,j] == grid_type['DYNAMIC_OCCUPIED']):
                    pygame.draw.rect(surface, color_dict['OCCUPIED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                elif(self.grid_map[i,j] == grid_type['UNOCCUPIED']):
                    pygame.draw.rect(surface, color_dict['UNOCCUPIED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
                elif(self.grid_map[i,j] == grid_type['UNEXPLORED']):
                    pygame.draw.rect(surface, color_dict['UNEXPLORED'], (self.x_scale * i, self.y_scale * j, self.x_scale, self.y_scale), 0)
class Raycast:
    #Pre-calculated values
    rad90deg = radians(90)
    rad270deg = radians(270)

    plane_width = None
    plane_height = None
    distance_to_plane = None
    center_x = None
    center_y = None

    strip_width = 10
    rays_number = None
    rays_angle = None

    world_elem_size = None

    view_angle_tan = None

    def __init__(self,plane_size, drone):
        self.FOV = radians(drone.yaw_range)
        self.depth = drone.yaw_depth
        self.initProjectionPlane(plane_size)

    def initProjectionPlane(self, plane_size):
        self.plane_width, self.plane_height = plane_size
        self.center_x = self.plane_width // 2
        self.center_y = self.plane_height // 2

        self.distance_to_plane = self.center_x / tan(self.FOV/2)

        self.rays_number = ceil(self.plane_width / self.strip_width)
        self.rays_angle = self.FOV / self.plane_width

        self.half_rays_number = self.rays_number//2


    def castRays(self, player, truth_grid_map, agents):
        rays = [self.castRay(player, pi*2 - radians(player.yaw), atan((-self.half_rays_number+i) * self.strip_width / self.distance_to_plane), truth_grid_map, agents) for i in range(self.rays_number)]
        hit_list = torch.zeros(len(agents), dtype=torch.int8)

        for ray in rays:
            hit_list = hit_list | ray['hit_list']
        for i, in_view in enumerate(hit_list):
            if in_view:
                agents[i].in_view = True
                agents[i].seen = True
            else:
                agents[i].in_view = False
        return rays

    
    def get_positive_angle(self, angle = None):

        angle = math.copysign((abs(angle) % (math.pi*2)), angle)
        if (angle < 0):
            angle += (math.pi*2)

        return angle

    def castRay(self, player, player_angle, ray_angle, truth_grid_map, agents):   
        x_step_size = truth_grid_map.x_scale - 1
        y_step_size = truth_grid_map.y_scale - 1
        # x_step_size = 1
        # y_step_size = 1
        
        ray_angle = player_angle + ray_angle

        dist = -1
        x_hit = -1
        y_hit = -1
        wall_hit = 0

        #Make shure angle between 0 and 2PI
        ray_angle = self.get_positive_angle(ray_angle)
        #Get directions which ray is faced
        faced_right = (ray_angle < self.rad90deg or ray_angle > self.rad270deg)
        faced_up = (ray_angle > pi)

        #Find Collision
        slope = tan(ray_angle)
        x = player.x
        y = player.y

        hit_list = torch.zeros(len(agents), dtype=torch.int8)

        if abs(slope) > 1:
            slope = 1 / slope

            y_step = -y_step_size if faced_up else y_step_size
            x_step = y_step * slope

            

            while (0 < x < truth_grid_map.dim[0] and 0 < y < truth_grid_map.dim[1]):
                i = int(x // truth_grid_map.x_scale)
                j = int(y // truth_grid_map.y_scale)

                for k, agent in enumerate(agents):
                    if (agent.position[0]-x)**2 + (agent.position[1]-y)**2 <= agent.radius**2:
                        x_hit = x
                        y_hit = y
                        hit_list[k] = 1
                if x_hit != -1:
                    break

                wall = truth_grid_map.grid_map[i, j]
                dist = (x - player.x)**2 + (y - player.y)**2
                if wall == 1 or dist >= self.depth**2:
                    x_hit = x
                    y_hit = y
                    wall_hit = wall
                    if wall == grid_type['OCCUPIED']:
                        player.map.grid_map[i, j] = grid_type['OCCUPIED']
                    break
                else:
                    player.map.grid_map[i, j] = grid_type['UNOCCUPIED']
                    # player.view_map[i, j] = 1
                x = x + x_step
                y = y + y_step
        
        else:
            x_step = x_step_size if faced_right else -x_step_size
            y_step = x_step * slope
            
            while (0 < x < truth_grid_map.dim[0] and 0 < y < truth_grid_map.dim[1]):
                i = int(x // truth_grid_map.x_scale)
                j = int(y // truth_grid_map.y_scale)

                for k, agent in enumerate(agents):
                    if (agent.position[0]-x)**2 + (agent.position[1]-y)**2 <= agent.radius**2:
                        x_hit = x
                        y_hit = y
                        hit_list[k] = 1
                if x_hit != -1:
                    break

                wall = truth_grid_map.grid_map[i, j]
                dist = (x-player.x)**2 + (y-player.y)**2
                if wall == 1 or dist >= self.depth**2:
                    x_hit = x
                    y_hit = y
                    wall_hit = wall
                    if wall == grid_type['OCCUPIED']:
                        player.map.grid_map[i, j] = grid_type['OCCUPIED']
                    break
                else:
                    player.map.grid_map[i, j] = grid_type['UNOCCUPIED']
                    # player.view_map[i, j] = 1

                x = x + x_step
                y = y + y_step
        result = {'coords':(x_hit,y_hit), 'wall':wall_hit, 'hit_list':hit_list}
        return result
class Drone2D():
    def __init__(self, init_x, init_y, init_yaw, dt, dim, params):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw % 360
        self.yaw_range = params.drone_view_range
        self.yaw_depth = params.drone_view_depth
        self.radius = params.drone_radius
        self.map = OccupancyGridMap(params.map_scale, dim, 0)
        # self.view_map = np.zeros((dim[0]//params.map_scale, dim[1]//params.map_scale))
        self.velocity = np.array([0, 0])
        self.dt = dt
        self.rays = {}
        self.raycast = Raycast(dim, self)
        self.params = params

    def step_yaw(self, action):
        # print(action)
        self.yaw = (self.yaw + action * self.dt) % 360

    def raycasting(self, gt_map, agents):
        # self.view_map = np.zeros_like(self.view_map)
        self.rays = self.raycast.castRays(self, gt_map, agents)

    def brake(self):
        if norm(self.velocity) <= self.params.drone_max_acceleration * self.dt:
            self.velocity = np.zeros(2)

        else:
            self.velocity = self.velocity - self.velocity / norm(self.velocity) * self.params.drone_max_acceleration * self.dt
            self.x += self.velocity[0] * self.dt
            self.y += self.velocity[1] * self.dt

    def is_collide(self, gt_map, agents):
        grid = gt_map.get_grid(self.x, self.y)
        if grid == grid_type['OCCUPIED']:
            # print("collision with static obstacles")
            return 1
        for agent in agents:
            if norm(agent.position - np.array([self.x, self.y])) < agent.radius + self.radius:
                # print("collision with dynamic obstacles")
                return 2

        return 0
    
    def get_local_map(self):
        drone_idx = (int(self.x // self.params.map_scale), int(self.y // self.params.map_scale))
        edge_len = 2 * (self.params.drone_view_depth // self.params.map_scale)
        local_map = np.pad(self.map.grid_map, ((edge_len,edge_len),(edge_len,edge_len)), 'constant', constant_values=0)
        return local_map[drone_idx[0] : drone_idx[0] + 2 * edge_len + 1, drone_idx[1] : drone_idx[1] + 2 * edge_len + 1]

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
        pygame.draw.circle(surface, (255,255,255), (self.x, self.y), self.radius)
class Waypoint2D(object):
    def __init__(self, pos=np.array([0,0]), vel=np.array([0,0])):
        self.position = pos
        self.velocity = vel
class Trajectory2D(object):
    def __init__(self):
        self.positions = []
        self.velocities = []
    def pop(self):
        self.positions.pop(0)
        self.velocities.pop(0)
    def __len__(self):
        return len(self.positions)

class Primitive_Node:
        def __init__(self, pos, vel, cost, target, parent_index, coeff, itr):
            self.position = pos  
            self.velocity = vel 
            self.cost = cost
            self.parent_index = parent_index
            self.coeff = coeff
            self.itr = itr
            self.total_cost = cost + 0.5*norm(pos-target) + 0.2*norm(vel)
            self.get_index()
        def __lt__(self, other_node):
            return self.total_cost < other_node.total_cost
        def get_index(self):
            self.index = (round(self.position[0])//10, round(self.position[1])//10, round(self.velocity[0])//2, round(self.velocity[1])//2)
class Primitive(object):
    def __init__(self, drone, params):
        self.params = params
        self.u_space = np.arange(-params.drone_max_acceleration, params.drone_max_acceleration, 10)
        # self.u_space = np.array([-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15])
        self.dt = 2
        self.sample_num = 10 # sampling number for collision check
        self.target = np.array([drone.x, drone.y])
        self.search_threshold = 10
        self.cost_ratio = 100

    def set_target(self, target_pos):
        self.target = target_pos

    def plan(self, start_pos, start_vel, occupancy_map, agents, update_t):
        """
        A star path search
        input:
            s_x: start x position 
            s_y: start y position 
            gx: goal x position 
            gy: goal y position
        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        start_node = Primitive_Node(pos=start_pos, 
                                    vel=start_vel,
                                    cost=0, 
                                    target=self.target,
                                    parent_index=-1,
                                    coeff=None,
                                    itr=0)

        open_set, closed_set = dict(), dict()
        open_set[start_node.index] = start_node
        itr = 0
        while 1:
            itr += 1
            if len(open_set) == 0 or itr >= 50:
                # print("No solution found in limitied time")
                goal_node = None
                success = False
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].total_cost)
            current = open_set[c_id]

            if norm(current.position - self.target) <= self.search_threshold:
                # print("Find goal")
                goal_node = current
                success = True
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            sub_node_list = [Primitive_Node(pos=np.around(np.array([1, self.dt, self.dt**2]) @ np.array([[current.position[0], current.position[1]], [current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]])), 
                                            vel=np.array([1, 2*self.dt]) @ np.array([[current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]]), 
                                            cost=current.cost + (x_acc**2 + y_acc**2)/self.cost_ratio + 10, 
                                            target=self.target,
                                            parent_index=current.index,
                                            coeff=np.array([[current.position[0], current.velocity[0], x_acc / 2], [current.position[1], current.velocity[1], y_acc / 2]]),
                                            itr = current.itr + 1) for x_acc in self.u_space 
                                                                   for y_acc in self.u_space 
                                                                   if self.is_free(np.array([[current.position[0], current.velocity[0], x_acc / 2], [current.position[1], current.velocity[1], y_acc / 2]]), occupancy_map, agents, current.itr) and 
                                                                      norm(np.array([1, 2*self.dt]) @ np.array([[current.velocity[0], current.velocity[1]], [x_acc/2, y_acc/2]])) < self.params.drone_max_speed]
            for next_node in sub_node_list:
                if next_node.index in closed_set:
                    continue

                if next_node.index not in open_set:
                    open_set[next_node.index] = next_node  # discovered a new node
                else:
                    if open_set[next_node.index].cost > next_node.cost:
                        # This path is the best until now. record it
                        open_set[next_node.index] = next_node
        # print("planning time:", time.time()-time1)
        trajectory = Trajectory2D()
        if success:
            cur_node = goal_node
            
            while(cur_node!=start_node): 
                trajectory.positions.extend([waypoint_from_traj(cur_node.coeff, t).position for t in np.arange(self.dt, 0, -update_t)])
                trajectory.velocities.extend([waypoint_from_traj(cur_node.coeff, t).velocity for t in np.arange(self.dt, 0, -update_t)])
                cur_node = closed_set[cur_node.parent_index]
            trajectory.positions.reverse()
            trajectory.velocities.reverse()

        return trajectory, success

    
    def is_free(self, coeff, occupancy_map, agents, itr):
        """Check if there is collision with the trajectory using sampling method

        Args:
            occupancy_map (_type_): Occupancy Map
            agents (_type_): Dynamic obstacles
        """
        for t in np.arange(0, self.dt, self.dt / self.sample_num):

            position = np.around(np.array([1, t, t**2]) @ coeff.T)

            grid = occupancy_map.get_grid(position[0] - self.params.drone_radius, position[1])
            if grid == 1:
                return False

            grid = occupancy_map.get_grid(position[0], position[1])
            if grid == 1:
                return False

            grid = occupancy_map.get_grid(position[0] + self.params.drone_radius, position[1])
            if grid == 1:
                return False

            grid = occupancy_map.get_grid(position[0], position[1] - self.params.drone_radius)
            if grid == 1:
                return False

            grid = occupancy_map.get_grid(position[0], position[1] + self.params.drone_radius)
            if grid == 1:
                return False

            for agent in agents:
                if agent.seen:
                    global_t = t + itr * self.dt
                    new_position = agent.estimate_pos + agent.estimate_vel * global_t
                    if norm(position - new_position) <= self.params.drone_radius + agent.radius:
                        return False


        return True
class Drone2DEnv1(gym.Env):
     
    def __init__(self, params):
        np.seterr(divide='ignore', invalid='ignore')
        self.params = params
        self.dt = params.dt

        # Setup pygame environment
        self.dim = params.map_size
        self.is_render = params.render
        if self.is_render:
            pygame.init()
            self.screen = pygame.display.set_mode(self.dim)
            self.clock = pygame.time.Clock()
        
        self.target_list = [np.array([520, 100]), np.array([120, 50]), np.array([120, 380]), np.array([520, 380])]
        random.shuffle(self.target_list)

        self.drone = Drone2D(self.dim[0] / 2, params.drone_radius + params.map_scale, -90, self.dt, self.dim, params)

        circular_obstacles = []
        for i in range(self.params.pillar_number):
            collision_free = False
            while not collision_free:
                obs = np.array([random.randint(50,self.dim[0]-50), random.randint(50,self.dim[1]-50), random.randint(30,50)])
                collision_free = True
                for target in self.target_list:
                    if norm(target - obs[:-1]) <= params.drone_radius + 20 + obs[-1]:
                        collision_free = False
                        break
                if norm(np.array([self.drone.x, self.drone.y]) - obs[:-1]) <= params.drone_radius + 10 + obs[-1]:
                    collision_free = False
                
            circular_obstacles.append(obs)

        self.obstacles = {
            'circular_obstacles'  : circular_obstacles,
            'rectangle_obstacles' : []
        }
        
        # Define workspace model for RVO model (approximate using circles)
        self.ws_model = obs_dict_to_ws_model(self.obstacles)
        
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
    
        self.planner = Primitive(self.drone, params)
        
        self.trajectory = Trajectory2D()
        self.swep_map = np.zeros([64, 48])
        self.state_machine = state_machine['WAIT_FOR_GOAL']
        self.state_changed = False
        self.failed_plan = 0

        # Define action and observation space
        self.info = {
            'drone':self.drone,
            'trajectory':self.trajectory,
            'state_machine':self.state_machine,
            'collision_flag':0
        }
        self.action_space = gym.spaces.Box(np.array([-1]), np.array([1]), shape=(1,))
        local_map_size = 4 * (params.drone_view_depth // params.map_scale) + 1
        self.observation_space = gym.spaces.Dict(
            {
                # 'local_map' : gym.spaces.MultiDiscrete(4*np.ones((1, local_map_size, local_map_size))),
                'yaw_angle' : gym.spaces.Box(np.array([0]), np.array([360]), shape=(1,), dtype=np.float32), 
                'local_map' : gym.spaces.Box(np.zeros((1, local_map_size, local_map_size)), 
                                                       4*np.ones((1, local_map_size,local_map_size)), 
                                                       shape=(1, local_map_size, local_map_size),
                                                       dtype=np.float32),
                'swep_map'  : gym.spaces.Box(np.zeros((1, local_map_size, local_map_size)), 
                                                       10*np.ones((1, local_map_size,local_map_size)), 
                                                       shape=(1, local_map_size, local_map_size),
                                                       dtype=np.float32)
            }
        )
    
    def step(self, a):
        done = False
        self.state_changed = False
        # Update state machine
        if self.state_machine == state_machine['GOAL_REACHED']:
            self.state_machine = state_machine['WAIT_FOR_GOAL']
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
        if self.state_machine == state_machine['WAIT_FOR_GOAL']:
            self.planner.set_target(self.target_list[-1])
            self.target_list.pop()
            self.state_machine = state_machine['PLANNING']
        # mouse = pygame.mouse.get_pressed()
        # if mouse[0]:
        #     success = False
        #     x, y = pygame.mouse.get_pos()
        #     self.planner.set_target(np.array([x, y]))
        #     self.trajectory, success = self.planner.plan(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.drone.map, self.agents, self.dt)
        #     self.state_changed = True
        #     if not success:
        #         self.drone.brake()
        #         self.state_machine = state_machine['PLANNING']
        #     else:
        #         self.state_machine = state_machine['EXECUTING']

        #Plan
        if self.state_machine == state_machine['PLANNING']:
            self.trajectory, success = self.planner.plan(np.array([self.drone.x, self.drone.y]), self.drone.velocity, self.drone.map, self.agents, self.dt)
            if not success:
                self.drone.brake()
                self.failed_plan += 1
                if self.failed_plan >= 3 and norm(self.drone.velocity)==0:
                    done = True
                # print("path not found, replanning")
            else:
                # print("path found")
                self.state_changed = True
                self.state_machine = state_machine['EXECUTING']
                self.failed_plan = 0

        # If collision detected for planned trajectory, replan
        swep_map = np.zeros_like(self.map_gt.grid_map)
        for i, pos in enumerate(self.trajectory.positions):
            swep_map[int(pos[0]//self.params.map_scale), int(pos[1]//self.params.map_scale)] = i * self.dt
            for agent in self.agents:
                if agent.seen:
                    estimate_pos = agent.estimate_pos + i * self.dt * agent.estimate_vel
                    if norm(estimate_pos - pos) <= self.drone.radius + agent.radius:
                        self.state_machine = state_machine['PLANNING']
                        self.state_changed = True   
        obs_map = np.where((self.drone.map.grid_map==0) | (self.drone.map.grid_map==2), 0, 1)
        if np.sum(obs_map * swep_map) > 0:
            self.state_machine = state_machine['PLANNING']
            self.state_changed = True

        # Execute trajectory
        if self.trajectory.positions != [] :
            self.drone.velocity = self.trajectory.velocities[0]
            self.drone.x = round(self.trajectory.positions[0][0])
            self.drone.y = round(self.trajectory.positions[0][1])
            self.trajectory.pop()
            if self.trajectory.positions == []:
                self.state_changed = True
                self.state_machine = state_machine['GOAL_REACHED']
        
        # Execute gaze control
        self.drone.step_yaw(a*self.params.drone_max_yaw_speed)
        
        # Print state machine
        # if self.state_changed:
        #     if self.state_machine == state_machine['GOAL_REACHED']:
        #         print("state: goal reached")
        #     elif self.state_machine == state_machine['WAIT_FOR_GOAL']:
        #         print("state: wait for goal")
        #     elif self.state_machine == state_machine['PLANNING']:
        #         print("state: planning")
        #     elif self.state_machine == state_machine['EXECUTING']:
        #         print("state: executing trajectory")

        # Return reward

        # lookahead: 1. 给速度方向，reward定为yaw和速度方向差距 2. 加入地图信息和中间reward（减弱地图信息干扰）
        collision_state = self.drone.is_collide(self.map_gt, self.agents)
        if collision_state == 1:
            if self.params.record and self.params.gaze_method != 'NoControl':
                pygame.image.save(self.screen, './experiment/fails/'+self.params.gaze_method+'_static_'+ str(datetime.now())+'.png')
            reward = -1000.0
            done = True
        elif collision_state == 2:
            if self.params.record and self.params.gaze_method != 'NoControl':
                pygame.image.save(self.screen, './experiment/fails/'+self.params.gaze_method+'_dynamic_'+ str(datetime.now())+'.png')
            reward = -1000.0
            done = True
        elif self.state_machine == state_machine['GOAL_REACHED']:
            reward = 100.0
            done = False
            if len(self.target_list) == 0:
                done = True
        x = np.arange(int(self.dim[0]//self.params.map_scale)).reshape(-1, 1) * self.params.map_scale
        y = np.arange(int(self.dim[1]//self.params.map_scale)).reshape(1, -1) * self.params.map_scale

        vec_yaw = np.array([math.cos(math.radians(self.drone.yaw)), -math.sin(math.radians(self.drone.yaw))])
        view_angle = math.radians(self.drone.yaw_range / 2)
        view_map = np.where(np.logical_or((self.drone.x - x)**2 + (self.drone.y - y)**2 <= 0, np.logical_and(np.arccos(((x - self.drone.x)*vec_yaw[0] + (y - self.drone.y)*vec_yaw[1]) / np.sqrt((self.drone.x - x)**2 + (self.drone.y - y)**2)) <= view_angle, ((self.drone.x - x)**2 + (self.drone.y - y)**2 <= self.drone.yaw_depth ** 2))), 1, 0)
        reward = float(np.sum(view_map * np.where(swep_map == 0, 0, 1)))
            
        # wrap up information
        self.info = {
            'drone':self.drone,
            'trajectory':self.trajectory,
            'swep_map':self.swep_map,
            'state_machine':self.state_machine,
            'collision_flag':collision_state
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

        return state, reward, done, self.info
    
    def reset(self):
        self.__init__(params=self.params)
        local_map_size = 4 * (self.params.drone_view_depth // self.params.map_scale) + 1
        return {'local_map' : self.drone.get_local_map()[None], 
                'swep_map'  : np.zeros((1, local_map_size, local_map_size)),
                'yaw_angle' : np.array([self.drone.yaw])}
        
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
            draw_static_obstacle(self.screen, self.obstacles, (200, 200, 200))
            
            if len(self.trajectory.positions) > 1:
                pygame.draw.lines(self.screen, (100,100,100), False, self.trajectory.positions)

            if len(self.agents) > 0:
                for agent in self.agents:
                    agent.render(self.screen)
            pygame.draw.circle(self.screen, (0,0,255), self.planner.target, self.drone.radius)
            default_font = pygame.font.SysFont('Arial', 15)
            pygame.Surface.blit(self.screen,
                default_font.render('STATE: '+list(state_machine.keys())[list(state_machine.values()).index(self.state_machine)], False, (0, 102, 0)),
                (0, 0)
            )
            
            pygame.display.update()
            self.clock.tick(60)
    
    