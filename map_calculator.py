import gym
import easydict
import main
import numpy as np
import pandas as pd
import time

from math import sin, cos, radians
from tqdm.contrib.itertools import product
from utils import *

def env_metrics(index):
    params = Params(agent_number=index['agent_number'],
                    agent_radius=index['agent_size'],
                    agent_max_speed=index['agent_speed'],
                    gaze_method='NoControl',
                    planner='NoMove')

    position_step = 60
    T = 12
    x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)
    total_survive = 0
    env = gym.make(params.env, params=params)
    for x in x_range:
        for y in y_range:
            env.reset()
            for t in np.arange(0, T, 0.1):
                env.drone.x = x
                env.drone.y = y
                _, _, done, info = env.step(0)
                env.render()
                if info['collision_flag'] == 2:
                    break
            total_survive += t

    return total_survive / (len(x_range) * len(y_range))

all_metrics = []

for map_id in range(5):

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
df.to_csv("./experiment/metrics/metrics_6m_12s.csv")