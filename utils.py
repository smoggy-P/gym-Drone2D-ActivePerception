import math
import numpy as np
from math import sin, cos
from numpy.linalg import norm

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
    