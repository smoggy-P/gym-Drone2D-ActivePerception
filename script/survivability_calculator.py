import gym
import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, '/home/cc/moji_ws/gym-Drone2D-ActivePerception')
import main

from math import sin, cos, radians
from tqdm.contrib.itertools import product
from utils import *

def env_metrics(index):
    params = Params(agent_number=index['agent_number'],
                    agent_radius=index['agent_size'],
                    agent_max_speed=index['agent_speed'],
                    motion_profile=index['motion_profile'],
                    map_id=index['map_id'],
                    gaze_method='NoControl',
                    planner='NoMove',
                    debug=True,
                    static_map='maps/empty_map.npy')
    params.render = False
    position_step = 60
    T = 24
    x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)
    
    collision_state = np.zeros((len(x_range), len(y_range), int(T/0.1)))
    
    env = gym.make(params.env, params=params)
    for x in x_range:
        for y in y_range:
            env.reset()
            for t in np.arange(0, T, 0.1):
                env.drone.x = x
                env.drone.y = y
                _, _, done, info = env.step(0)
                # env.render()
                if info['collision_flag'] == 2:
                    collision_state[int((x - params.map_scale - params.drone_radius)/position_step), int((y - params.map_scale - params.drone_radius)/position_step), int(t/0.1)] = 1

    return collision_state

all_metrics = []
collision_states = []
for map_id in range(20):
    for (agent_num, agent_size, agent_vel) in product([10, 20, 30], [5, 10, 15], [20, 40, 60]):
        index = {'motion_profile':'CVM',
                'pillar_number':0,
                'agent_number':agent_num,
                'agent_speed':agent_vel,
                'agent_size':agent_size,
                'map_id':map_id}
        collision_state = env_metrics(index)
        collision_states.append(collision_state)
    
np.save("collision_states_6m_12s_RVO.npy", np.array(collision_states))

# metric_dict = {"metric":all_metrics}
# df = pd.DataFrame(metric_dict)
# df.to_csv("metrics_6m_12s_RVO.csv")