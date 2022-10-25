import numpy as np
import pygame
from math import cos, sin, atan2, asin, sqrt
from utils import check_in_view
from math import pi as PI

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
        self.var = None
        self.estimate_vel = None
        self.estimate_pos = None
        
    def step(self, edge_size_x, edge_size_y, map_width, map_height, drone, dt):
        print(self.velocity)
        new_position = self.position + np.array(self.velocity) * dt
            
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
        if check_in_view(drone, self.position):
            if not self.seen:
                self.seen = True
            self.var = 0
            self.estimate_vel = self.velocity
            self.estimate_pos = self.position
        else:
            if self.seen:
                self.var += self.dt*10

        if self.seen:
            self.estimate_pos = self.estimate_pos + self.estimate_vel * dt
        
    def render(self, surface):
        if self.seen:
            pygame.draw.circle(surface, pygame.Color(0, 250, 250), np.rint(self.position).astype(int), int(round(self.radius)), 0)
            pygame.draw.circle(surface, pygame.Color(250, 0, 0), np.rint(self.estimate_pos).astype(int), int(round(self.radius+self.var)), 1)
            pygame.draw.line(surface, pygame.Color(250, 0, 0), np.rint(self.position).astype(int), np.rint(self.estimate_pos).astype(int), 1)
        else:
            pygame.draw.circle(surface, pygame.Color(250, 0, 0), np.rint(self.position).astype(int), int(round(self.radius)), 0)
        pygame.draw.line(surface, pygame.Color(0, 255, 0), np.rint(self.position).astype(int), np.rint((self.position + self.velocity)).astype(int), 1)

def distance(pose1, pose2):
    """ compute Euclidean distance for 2D """
    return sqrt((pose1[0]-pose2[0])**2+(pose1[1]-pose2[1])**2)+0.001


def RVO_update(agents, ws_model):
    """ compute best velocity given the desired velocity, current velocity and workspace model"""
    
    X = [agent.position for agent in agents]
    V_des = [agent.pref_velocity for agent in agents]
    V_current = [agent.velocity for agent in agents]
    
    ROB_RAD = agents[0].radius+0.1
       
    for i in range(len(X)):
        RVO_BA_all = [[[X[i][0]+0.5*(V_current[j][0]+V_current[i][0]), X[i][1]+0.5*(V_current[j][1]+V_current[i][1])], 
                      [cos(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) + asin(2*ROB_RAD/distance(X[i], X[j]))), sin(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) + asin(2*ROB_RAD/distance(X[i], X[j])))],
                      [cos(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) - asin(2*ROB_RAD/distance(X[i], X[j]))), sin(atan2(X[j][1]-X[i][1], X[j][0]-X[i][0]) - asin(2*ROB_RAD/distance(X[i], X[j])))],
                      distance(X[i], X[j]),
                      2.2*ROB_RAD] for j in range(len(X)) if j != i] + [[X[i], 
                                                                        [cos(atan2(hole[1]-X[i][1], hole[0]-X[i][0]) + asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2]))), sin(atan2(hole[1]-X[i][1], hole[0]-X[i][0]) + asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2])))], 
                                                                        [cos(atan2(hole[1]-X[i][1], hole[0]-X[i][0])-asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2]))), sin(atan2(hole[1]-X[i][1], hole[0]-X[i][0])-asin((hole[2]*1+ROB_RAD)/distance(X[i], hole[0:2])))], 
                                                                        distance(X[i], hole[0:2]), 
                                                                        hole[2]*1.5+ROB_RAD] for hole in ws_model['circular_obstacles']]
        vA_post = intersect(X[i], V_des[i], RVO_BA_all)
        agents[i].velocity = np.array(vA_post[:])


def intersect(pA, vA, RVO_BA_all):
    # print '----------------------------------------'
    # print 'Start intersection test'
    norm_v = distance(vA, [0, 0])
    suitable_V = []
    unsuitable_V = []
    for theta in np.arange(0, 2*PI, 0.5):
        for rad in np.arange(0.02, norm_v+0.02, norm_v/5.0):
            new_v = [rad*cos(theta), rad*sin(theta)]
            suit = True
            for RVO_BA in RVO_BA_all:
                p_0 = RVO_BA[0]
                left = RVO_BA[1]
                right = RVO_BA[2]
                dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
                theta_dif = atan2(dif[1], dif[0])
                theta_right = atan2(right[1], right[0])
                theta_left = atan2(left[1], left[0])
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
        p_0 = RVO_BA[0]
        left = RVO_BA[1]
        right = RVO_BA[2]
        dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
        theta_dif = atan2(dif[1], dif[0])
        theta_right = atan2(right[1], right[0])
        theta_left = atan2(left[1], left[0])
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
            p_0 = RVO_BA[0]
            left = RVO_BA[1]
            right = RVO_BA[2]
            dif = [new_v[0]+pA[0]-p_0[0], new_v[1]+pA[1]-p_0[1]]
            theta_dif = atan2(dif[1], dif[0])
            theta_right = atan2(right[1], right[0])
            theta_left = atan2(left[1], left[0])
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
    if abs(theta_right - theta_left) <= PI:
        if theta_right <= theta_dif <= theta_left:
            return True
        else:
            return False
    else:
        if (theta_left <0) and (theta_right >0):
            theta_left += 2*PI
            if theta_dif < 0:
                theta_dif += 2*PI
            if theta_right <= theta_dif <= theta_left:
                return True
            else:
                return False
        if (theta_left >0) and (theta_right <0):
            theta_right += 2*PI
            if theta_dif < 0:
                theta_dif += 2*PI
            if theta_left <= theta_dif <= theta_right:
                return True
            else:
                return False

def compute_V_des(X, goal, V_max):
    V_des = []
    for i in range(len(X)):
        dif_x = [goal[i][k]-X[i][k] for k in range(2)]
        norm = distance(dif_x, [0, 0])
        norm_dif_x = [dif_x[k]*V_max[k]/norm for k in range(2)]
        V_des.append(norm_dif_x[:])
        if reach(X[i], goal[i], 0.1):
            V_des[i][0] = 0
            V_des[i][1] = 0
    return V_des
            
def reach(p1, p2, bound=0.5):
    if distance(p1,p2)< bound:
        return True
    else:
        return False
    
    
