import numpy as np
import pygame
import math
import torch

from numpy import array
from math import atan2, asin, cos, sin, radians, tan, pi, ceil, degrees
from numpy.linalg import norm


grid_type = {
    'DYNAMIC_OCCUPIED' : 3,
    'OCCUPIED' : 1,
    'UNOCCUPIED' : 2,
    'UNEXPLORED' : 0
}

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

def draw_cov(surface, mean, cov):
    # Compute the eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    major_axis = 2 * np.sqrt(5.991 * eigenvalues[0])  # 95% confidence interval for major axis
    minor_axis = 2 * np.sqrt(5.991 * eigenvalues[1])  # 95% confidence interval for minor axis
    angle = degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))  # Angle between major axis and x-axis

    target_rect = pygame.Rect((int(mean[0]-major_axis/2-2), int(mean[1]-minor_axis/2-2), int(major_axis+4), int(minor_axis+4)))
    shape_surf = pygame.Surface(target_rect.size, pygame.SRCALPHA)
    pygame.draw.ellipse(shape_surf, (255, 255, 0), (0, 0, *target_rect.size), 1)
    rotated_surf = pygame.transform.rotate(shape_surf, angle)
    surface.blit(rotated_surf, rotated_surf.get_rect(center = target_rect.center))

class Params: 
    def __init__(self, env='gym-2d-perception-v2', debug=True, record_img=False, 
                 trained_policy=False, policy_dir='./trained_policy/lookahead.zip', dt=0.1, 
                 map_scale=10, map_size=[500,500], agent_radius=10, drone_max_acceleration=40, 
                 drone_radius=10, drone_max_yaw_speed=80, drone_view_depth=80, drone_view_range=90, 
                 img_dir='./', max_flight_time=80, gaze_method='LookAhead', planner='Primitive', var_cam=0, 
                 drone_max_speed=40, motion_profile='CVM', pillar_number=0, agent_number=10, 
                 agent_max_speed=40, map_id=0, init_pos=[50, 50], target_list=[[50, 460]], static_map='maps/empty_map.npy'):
        
        self.env = env
        if debug:
            self.render = True
            self.record = False
        else:
            self.render = False
            self.record = True
        self.record_img = record_img
        self.trained_policy = trained_policy
        self.policy_dir = policy_dir
        self.dt = dt
        self.map_scale = map_scale
        self.map_size = map_size
        self.agent_radius = agent_radius
        self.drone_max_acceleration = drone_max_acceleration
        self.drone_radius = drone_radius
        self.drone_max_yaw_speed = drone_max_yaw_speed
        self.drone_view_depth = drone_view_depth
        self.drone_view_range = drone_view_range
        self.img_dir = img_dir
        self.max_flight_time = max_flight_time
        self.gaze_method = gaze_method
        self.planner = planner
        self.var_cam = var_cam
        self.drone_max_speed = drone_max_speed
        self.motion_profile = motion_profile
        self.pillar_number = pillar_number
        self.agent_number = agent_number
        self.agent_max_speed = agent_max_speed
        self.map_id = map_id
        self.init_position = init_pos
        self.target_list = target_list
        self.static_map = static_map
class KalmanFilter:

    def copy(self):
        new_filter = KalmanFilter(params=self.params)
        new_filter.mu_upds = self.mu_upds
        new_filter.Sigma_upds = self.Sigma_upds
        new_filter.ts = self.ts
        return new_filter

    def __init__(self, params, mu=np.zeros([4,1]), Sigma=np.diag([1, 1, 10, 10])):
        self.active = False
        self.params = params
        
        # check that initial state makes sense
        Dx = mu.shape[0]
        assert mu.shape == (Dx, 1)
        assert Sigma.shape == (Dx, Dx)

        self.mu_upds = []
        self.Sigma_upds = []

        self.ts = []

        self.mu_upds.append(mu)
        self.Sigma_upds.append(Sigma)
        self.ts.append(0.) # this is time t = 0
            
        # the dimensionality of the state vector
        self.Dx = Dx
    
        noise_var_x_pos = 0.1 if params.var_cam != 0 else 0.001 # variance of spatial process noise
        noise_var_x_vel = 0.1 if params.var_cam != 0 else 0.001  # variance of velocity process noise
        noise_var_z = params.var_cam # variance of measurement noise for z_x and z_y

        self.F = np.array([[1,0,0.1,0  ],
                           [0,1,0  ,0.1],
                           [0,0,1  ,0  ],
                           [0,0,0  ,1  ]], dtype=np.float64) 
        self.H = np.array([[1,0,0,0],
                           [0,1,0,0]], dtype=np.float64) 
        self.Sigma_x = np.array([[noise_var_x_pos,0,0,0],
                                 [0,noise_var_x_pos,0,0],
                                 [0,0,noise_var_x_vel,0],
                                 [0,0,0,noise_var_x_vel]], dtype=np.float64)  
        self.Sigma_z = np.array([[noise_var_z,0],
                                 [0,noise_var_z]], dtype=np.float64)  
    
    def estimate_pos(self, t):
        cur_pos = self.mu_upds[-1][:2,0]
        cur_vel = self.mu_upds[-1][2:,0]
        return cur_pos + t * cur_vel

    def predict(self):
        mu_prev = self.mu_upds[-1]
        Sigma_prev = self.Sigma_upds[-1]
        t = self.ts[-1]
        mu = self.F.dot(mu_prev)
        Sigma = self.F.dot(Sigma_prev).dot(self.F.T) + self.Sigma_x
        self.mu_upds.append(mu)
        self.Sigma_upds.append(Sigma)
        self.ts.append(t + 1)
        achieved_filter = None
        if Sigma[0,0] >= 150 or \
           (not(10 + self.params.agent_radius < mu[0] < self.params.map_size[0] - 10 - self.params.agent_radius)) or \
           (not(10 + self.params.agent_radius < mu[1] < self.params.map_size[1] - 10 - self.params.agent_radius)):
            achieved_filter = self.copy()
            self.__init__(self.params)
        return achieved_filter
    
    def update(self, z):
        achieved_list = []
        # Object has been tracked
        if self.active:
            achieved_filter = self.predict()
            if not(achieved_filter is None):
                achieved_list.append(achieved_filter)
            if not(z is None):
                mu = self.mu_upds[-1]
                Sigma = self.Sigma_upds[-1]
                z = z.reshape(-1,1)

                S = self.Sigma_z + self.H.dot(Sigma).dot(self.H.T)
                K = Sigma.dot(self.H.T).dot(np.linalg.inv(S))
                
                mu_upd = mu + K.dot(z - self.H.dot(mu))
                Sigma_upd = (np.eye(4) - K.dot(self.H)).dot(Sigma)
                self.mu_upds[-1] = mu_upd
                self.Sigma_upds[-1] = Sigma_upd
        
        # Object is tracked for the first time
        elif not (z is None):
            self.__init__(self.params, mu=np.vstack([z.reshape(-1, 1), np.zeros([2,1])]), Sigma=np.diag([1, 1, 10, 10]))
            self.active = True
        
        return achieved_list
class Waypoint2D(object):
    def __init__(self, pos=np.array([0,0]), vel=np.array([0,0])):
        self.position = pos
        self.velocity = vel
class Trajectory2D(object):
    def __init__(self):
        self.positions = []
        self.velocities = []
        self.accelerations = []
    def pop(self):
        self.positions.pop(0)
        self.velocities.pop(0)
        self.accelerations.pop(0)
    def clear(self):
        self.positions = []
        self.velocities = []
        self.accelerations = []
    def __len__(self):
        return len(self.positions)
    def get_swep_map(self, swep_map):
        for i, pos in enumerate(self.positions):
            # unparam
            swep_map[int(pos[0]//10), int(pos[1]//10)] = i * 0.1
class RVO():
    @classmethod
    def RVO_update(self, agents, circular_obstacles):
        """ compute best velocity given the desired velocity, current velocity and workspace model"""
        
        X = [agent.position for agent in agents]
        V_des = [agent.pref_velocity for agent in agents]
        V_current = [agent.velocity for agent in agents]
        
        ROB_RAD = agents[0].radius+0.01
        
        for i in range(len(X)):
            try:
                vA = array([V_current[i][0], V_current[i][1]])
                pA = array([X[i][0], X[i][1]])
                RVO_BA_all = []
                for j in range(len(X)):
                    if i!=j:
                        vB = array([V_current[j][0], V_current[j][1]])
                        pB = array([X[j][0], X[j][1]])
                        # use RVO
                        transl_vB_vA = [pA[0]+0.5*(vB[0]+vA[0]), pA[1]+0.5*(vB[1]+vA[1])]
                        # use VO
                        # transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
                        dist_BA = norm(pA - pB)
                        theta_BA = atan2(pB[1]-pA[1], pB[0]-pA[0])
                        if 2*ROB_RAD > dist_BA:
                            dist_BA = 2*ROB_RAD
                        theta_BAort = asin(2*ROB_RAD/dist_BA)
                        theta_ort_left = theta_BA+theta_BAort
                        bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                        theta_ort_right = theta_BA-theta_BAort
                        bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                        RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, 2*ROB_RAD]
                        RVO_BA_all.append(RVO_BA)                
                for hole in circular_obstacles:
                    # hole = [x, y, rad]
                    vB = [0, 0]
                    pB = hole[0:2]
                    transl_vB_vA = [pA[0]+vB[0], pA[1]+vB[1]]
                    dist_BA = norm(pA - pB)
                    theta_BA = atan2(pB[1]-pA[1], pB[0]-pA[0])
                    # over-approximation of square to circular
                    OVER_APPROX_C2S = 1.5
                    rad = hole[2]*OVER_APPROX_C2S
                    if (rad+ROB_RAD) > dist_BA:
                        dist_BA = rad+ROB_RAD
                    theta_BAort = asin((rad+ROB_RAD)/dist_BA)
                    theta_ort_left = theta_BA+theta_BAort
                    bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                    theta_ort_right = theta_BA-theta_BAort
                    bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                    RVO_BA = [transl_vB_vA, bound_left, bound_right, dist_BA, rad+ROB_RAD]
                    RVO_BA_all.append(RVO_BA)
            except:
                return False
            vA_post = self.intersect(X[i], V_des[i], RVO_BA_all)
            agents[i].velocity = np.array(vA_post[:])
        return True

    @classmethod
    def intersect(self, pA, vA, RVO_BA_all):
        # print '----------------------------------------'
        # print 'Start intersection test'
        norm_v = norm(vA - [0, 0])
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
                    if self.in_between(theta_right, theta_dif, theta_left):
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
            if self.in_between(theta_right, theta_dif, theta_left):
                suit = False
                break
        if suit:
            suitable_V.append(new_v)
        else:
            unsuitable_V.append(new_v)
        #----------------------        
        if suitable_V:
            # print 'Suitable found'
            vA_post = min(suitable_V, key = lambda v: norm(v - vA))
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
                    dif = array([unsuit_v[0]+pA[0]-p_0[0], unsuit_v[1]+pA[1]-p_0[1]])
                    theta_dif = atan2(dif[1], dif[0])
                    theta_right = atan2(right[1], right[0])
                    theta_left = atan2(left[1], left[0])
                    if self.in_between(theta_right, theta_dif, theta_left):
                        small_theta = abs(theta_dif-0.5*(theta_left+theta_right))
                        if abs(dist*sin(small_theta)) >= rad:
                            rad = abs(dist*sin(small_theta))
                        big_theta = asin(abs(dist*sin(small_theta))/rad)
                        dist_tg = abs(dist*cos(small_theta))-abs(rad*cos(big_theta))
                        if dist_tg < 0:
                            dist_tg = 0                    
                        tc_v = dist_tg/norm(dif)
                        tc.append(tc_v)
                tc_V[tuple(unsuit_v)] = min(tc)+0.001
            WT = 0.2
            vA_post = min(unsuitable_V, key = lambda v: ((WT/tc_V[tuple(v)])+norm(v - vA)))
        return vA_post 
    
    @classmethod
    def in_between(self, theta_right, theta_dif, theta_left):
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
    def __init__(self, position, velocity, radius, max_speed, pref_velocity, group_id=0):
        super(Agent, self).__init__()
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.radius = radius
        self.max_speed = max_speed
        self.pref_velocity = np.array(pref_velocity)
        self.group_id = group_id
        
    def step(self, edge_size_x, edge_size_y, map_width, map_height, dt):
        new_position = self.position + self.velocity * dt
        
        # If stuck, change direction
        if norm(self.velocity) <= 5:
            self.pref_velocity = (array([[cos(pi/6), sin(-pi/6)],[sin(pi/6), cos(pi/6)]]) @ (self.pref_velocity.reshape(-1, 1))).flatten()

        # Change reference velocity if reaching the boundary
        if new_position[0] < edge_size_x + self.radius:
            self.pref_velocity[0] = abs(self.pref_velocity[0])
            
        elif new_position[0] > map_width - edge_size_x - self.radius:
            self.pref_velocity[0] = -abs(self.pref_velocity[0])
            
        if new_position[1] < edge_size_y + self.radius:
            self.pref_velocity[1] = abs(self.pref_velocity[1])
            
        elif new_position[1] > map_height - edge_size_y - self.radius:
            self.pref_velocity[1] = -abs(self.pref_velocity[1])
            
            
        self.position = self.position + np.array(self.velocity) * dt
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
        
    def init_obstacles(self, obstacles, agents):
        # Mark edges in Grid Map
        self.grid_map[0,:] = grid_type['OCCUPIED']
        self.grid_map[-1,:] = grid_type['OCCUPIED']
        self.grid_map[:,0] = grid_type['OCCUPIED']
        self.grid_map[:,-1] = grid_type['OCCUPIED']
        
        # Mark static obstacles in Grid Map
        for i in range(self.grid_map.shape[0]):
            for j in range(self.grid_map.shape[1]):
                for circle in obstacles:
                    if norm(self.get_real_pos(i,j) - np.array([circle[0], circle[1]])) <= circle[2]:
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

    def __init__(self,plane_size, drone, sigma):
        self.FOV = radians(drone.yaw_range)
        self.depth = drone.yaw_depth
        self.sigma = sigma
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
        rays = [self.castRay(player, pi*2 - radians(player.yaw), -self.FOV/2 + self.FOV/self.rays_number*i, truth_grid_map, agents) for i in range(self.rays_number)]
        hit_list = torch.zeros(len(agents), dtype=torch.int8)
        newly_tracked = 0

        for ray in rays:
            hit_list = hit_list | ray['hit_list']

        measurements = [None for i in range(len(agents))]

        for i, in_view in enumerate(hit_list):
            if in_view:
                measurements[i] = agents[i].position + self.sigma * np.random.randn(2)
                if player.trackers[i].active == False:
                    newly_tracked += 1
        
        return rays, newly_tracked, measurements

    
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
    def __init__(self, init_x, init_y, init_yaw, dt, params):
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw % 360
        self.yaw_range = params.drone_view_range
        self.yaw_depth = params.drone_view_depth
        self.radius = params.drone_radius
        self.map = OccupancyGridMap(params.map_scale, params.map_size, 0)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.dt = dt
        self.rays = {}
        self.raycast = Raycast(params.map_size, self, params.var_cam)
        self.params = params

        # Max tracking number is 100
        self.trackers = [KalmanFilter(params) for i in range(200)]

    def step_pos(self, trajectory):
        if trajectory.positions != []:
            self.acceleration = trajectory.accelerations[0]
            self.velocity = trajectory.velocities[0]
            self.x = round(trajectory.positions[0][0])
            self.y = round(trajectory.positions[0][1])
            trajectory.pop()

    def step_yaw(self, action):
        # print(action)
        self.yaw = (self.yaw + action * self.dt) % 360

    def get_measurements(self, gt_map, agents):
        self.rays, newly_tracked, measurements = self.raycast.castRays(self, gt_map, agents)
        return newly_tracked, measurements

    def update_tracker(self, measurements):
        achieved_list = []
        for i, measurement in enumerate(measurements):
            achieved_list.extend(self.trackers[i].update(measurement))
        return achieved_list

    def brake(self):
        if norm(self.velocity) <= self.params.drone_max_acceleration * self.dt:
            self.velocity = np.zeros(2)

        else:
            self.velocity = self.velocity - self.velocity / norm(self.velocity) * self.params.drone_max_acceleration * self.dt
            self.x += self.velocity[0] * self.dt
            self.y += self.velocity[1] * self.dt

    def is_collide(self, gt_map, agents):
        position = np.array([self.x, self.y])
        offsets = [(-self.params.drone_radius, 0), (0, 0), (self.params.drone_radius, 0), (0, -self.params.drone_radius), (0, self.params.drone_radius)]

        for offset in offsets:
            grid = gt_map.get_grid(position[0] + offset[0], position[1] + offset[1])
            if grid == 1:
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
                        (100,100,100), 
                        [self.x - self.yaw_depth,
                            self.y - self.yaw_depth,
                            2 * self.yaw_depth,
                            2 * self.yaw_depth], 
                        math.radians(self.yaw - self.yaw_range/2), 
                        math.radians(self.yaw + self.yaw_range/2),
                        2)
        angle1 = math.radians(self.yaw + self.yaw_range/2)
        angle2 = math.radians(self.yaw - self.yaw_range/2)
        pygame.draw.line(surface, (100,100,100), (self.x, self.y), (self.x + self.yaw_depth * cos(angle1), self.y - self.yaw_depth * sin(angle1)), 2)
        pygame.draw.line(surface, (100,100,100), (self.x, self.y), (self.x + self.yaw_depth * cos(angle2), self.y - self.yaw_depth * sin(angle2)), 2)
        pygame.draw.circle(surface, (100,100,100), (self.x, self.y), self.radius)