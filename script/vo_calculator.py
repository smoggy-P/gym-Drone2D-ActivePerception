import numpy as np
import main
import gym
import pandas as pd
from numpy.linalg import norm
from math import atan2, asin, cos, sin
from utils import Params
from tqdm.contrib.itertools import product

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


def env_metrics(index):

    params = Params(env='gym-2d-perception-v2',
                    agent_number=index['agent_number'],
                    agent_radius=index['agent_size'],
                    agent_max_speed=index['agent_speed'],
                    map_id=index['map_id'],
                    gaze_method='NoControl',
                    planner='NoMove',
                    drone_radius=0,
                    debug=True,
                    static_map='maps/empty_map.npy')
    params.render = False
    
    env = gym.make(params.env, params=params)
    env.reset()
    # env.render()

    agents = env.agents
    v_max = 60
    v_min = 20
    rA = 5
    position_step = 60

    x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)

    rates = []

    for x in x_range:
        for y in y_range:
            env.reset()
            pA = np.array([x, y])
            X = [agent.position for agent in agents]
            V_current = [agent.pref_velocity for agent in agents]
            rs = [agent.radius for agent in agents]
            already_collision = False
            RVO_BA_all = []
            for j in range(len(X)):
                vB = np.array([V_current[j][0], V_current[j][1]])
                pB = np.array([X[j][0], X[j][1]])
                rB = rs[j]

                # use VO
                dist_BA = norm(pA - pB)
                theta_BA = atan2(pB[1]-pA[1], pB[0]-pA[0])

                if dist_BA < rA+rB:
                    already_collision = True
                    break
            
                theta_BAort = asin((rA+rB)/dist_BA)
                theta_ort_left = theta_BA+theta_BAort
                bound_left = [cos(theta_ort_left), sin(theta_ort_left)]
                theta_ort_right = theta_BA-theta_BAort
                bound_right = [cos(theta_ort_right), sin(theta_ort_right)]
                RVO_BA = [vB, bound_left, bound_right, dist_BA, rA+rB]
                RVO_BA_all.append(RVO_BA)                

            if already_collision:
                rates.append(0)
                continue

            suitable_V = []
            unsuitable_V = []
            for theta in np.arange(0, 2*3.14, 0.2):
                for rad in np.arange(v_min, v_max, (v_max-v_min)/5.0):
                    new_v = [rad*cos(theta), rad*sin(theta)]
                    suit = True
                    for RVO_BA in RVO_BA_all:
                        theta_dif = atan2(new_v[1]-RVO_BA[0][1], new_v[0]-RVO_BA[0][0])
                        theta_right = atan2(RVO_BA[2][1], RVO_BA[2][0])
                        theta_left = atan2(RVO_BA[1][1], RVO_BA[1][0])
                        if in_between(theta_right, theta_dif, theta_left):
                            suit = False
                            break
                    if suit:
                        suitable_V.append(new_v)
                    else:
                        unsuitable_V.append(new_v)                
            rates.append(len(suitable_V)/(len(suitable_V)+len(unsuitable_V)))
    # print(V_current)
    # print("rates:", rates)
    print("mean:", np.mean(rates))
    return np.mean(rates)

all_metrics = []
for map_id in range(20):
    env_metric = []
    for (agent_num, agent_size, agent_vel) in product([10, 20, 30], [5, 10, 15], [20, 40, 60]):
        index = {'motion_profile':'CVM',
                'pillar_number':0,
                'agent_number':agent_num,
                'agent_speed':agent_vel,
                'agent_size':agent_size,
                'map_id':map_id}
        env_metric.append(env_metrics(index))
    all_metrics.append(env_metric)
metric_dict = {"metric":all_metrics}
df = pd.DataFrame(metric_dict)
df.to_csv('vo.csv')