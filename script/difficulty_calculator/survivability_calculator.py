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
    T = 12
    x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)
    
    
    survive_times = np.ones((len(x_range), len(y_range))) * T
    env = gym.make(params.env, params=params)
    env.reset()
    _, _, done, info = env.step(0)
    for t in np.arange(0, T, 0.1):
        for x in x_range:
            for y in y_range:
                drone_pos = np.array([x, y])
                for agent in env.agents:
                    if norm(agent.position - drone_pos) < agent.radius + env.drone.radius:
                        survive_times[x_range.index(x), y_range.index(y)] = min(t, survive_times[x_range.index(x), y_range.index(y)])
        _, _, done, info = env.step(0)
        # env.render()

    
    survive_times = survive_times - 0.1
    survive_times[survive_times < 0] = 0
    print(np.mean(survive_times))
    return np.mean(survive_times)

all_metrics = []

for map_id in [0]:

    env_metric = []
    for (agent_num, agent_size, agent_vel) in product(range(10, 30, 2), range(5, 15), range(20, 60, 5)):
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
df.to_csv("metrics_fit.csv")