import gym
import easydict
import main
import numpy as np
import pandas as pd

from math import sin, cos, radians
from tqdm.contrib.itertools import product

def env_metrics(index):

    params = easydict.EasyDict({
            'env':'gym-2d-perception-v2',
            'render':True,
            'record': False,

            'record_img': False,
            'trained_policy':False,
            'policy_dir':'./trained_policy/lookahead.zip',
            'dt':0.1,
            'map_scale':10,
            'map_size':[480,640],
            'agent_radius':10,
            'drone_max_acceleration':40,
            'drone_radius':10,
            'drone_max_yaw_speed':80,
            'drone_view_depth' : 80,
            'drone_view_range': 90,
            'img_dir':'./',
            'max_flight_time': 80,
            

            'gaze_method':'NoControl',#5
            'planner':'NoMove',#3

            'var_cam': 0,
            'drone_max_speed':40,#3

            'motion_profile':index['motion_profile'],
            'pillar_number':index['pillar_number'],
            'agent_number':index['agent_number'],
            'agent_max_speed':index['agent_speed'],
            'map_id':index['map_id']
        })

    position_step = 60
    T = 4
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
                if done:
                    break
            total_survive += t

    return total_survive / (len(x_range) * len(y_range))

all_metrics = []

for map_id in range(5):

    env_metric = []
    for (agent_num, agent_vel) in product([10, 20, 30], [20, 40, 60]):
        index = {'motion_profile':'CVM',
                'pillar_number':0,
                'agent_number':agent_num,
                'agent_speed':agent_vel,
                'map_id':map_id}
        survive_time = env_metrics(index)
        env_metric.append(survive_time)
    all_metrics.append(env_metric)

metric_dict = {"metric":all_metrics}
df = pd.DataFrame(metric_dict)
df.to_csv("./experiment/metrics/metrics_6m_4s.csv")