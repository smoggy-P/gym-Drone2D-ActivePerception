import gym
import numpy as np
import pandas as pd
import time
import sys
sys.path.insert(0, '/home/smoggy/thesis/gym-Drone2D-ActivePerception')
import main

from math import sin, cos, radians
from tqdm.contrib.itertools import product
from utils import *

def env_metrics(index):
    T = 12
    params = Params(env='gym-metric-v1',
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
    a = env.get_traversibility([5, 10, 15, 20, 25, 30, 35, 40, 45])
    print(a)

    return a

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
        survive_time = env_metrics(index)
        env_metric.append(survive_time)
    all_metrics.append(env_metric)

metric_dict = {"metric":all_metrics}
df = pd.DataFrame(metric_dict)
df.to_csv('traversibility.csv')