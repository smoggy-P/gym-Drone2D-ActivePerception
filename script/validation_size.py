import gym
import numpy as np
import pandas as pd
import random
import os
import warnings
from yaw_planner import Oxford, LookAhead, NoControl, Rotating, Owl, LookGoal
from datetime import datetime
from utils import *
from itertools import product
from tqdm.contrib.itertools import product as tqdm_product
from experiment import Experiment
warnings.filterwarnings("ignore", category=DeprecationWarning) 
result_dir = './experiment/results_'+str(datetime.now())+'.csv'
metric_dir = './experiment/metrics/'

# Environment difficulty
agent_numbers = [10, 20, 30]
agent_speeds = [20, 40, 60]
map_ids = range(5)
env_params = tqdm_product(agent_numbers, agent_speeds, map_ids)

# Problem difficulty
drone_max_speeds = [20, 40, 60]

all_metrics = []

for (agent_number, agent_vel, map_id) in env_params:

    params = Params(debug=True, 
                    map_id=map_id,
                    gaze_method='NoControl',
                    planner='NoMove',
                    agent_number=agent_number,
                    agent_max_speed=agent_vel,
                    agent_radius=-1)
    params.render = False
    env = gym.make('gym-2d-perception-v2', params=params)
    
    # Calculate difficulty
    # position_step = 60
    # T = 12
    # x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    # y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)
    # total_survive = 0
    # env = gym.make(params.env, params=params)
    # for x in x_range:
    #     for y in y_range:
    #         # Give random velocity to agents
    #         env.reset()
    #         for t in np.arange(0, T, 0.1):
    #             env.drone.x = x
    #             env.drone.y = y
    #             _, _, done, info = env.step(0)
    #             # env.render()
    #             if info['collision_flag'] == 2:
    #                 break
    #         total_survive += t

    # print(total_survive / (len(x_range) * len(y_range)))
    # all_metrics.append(total_survive / (len(x_range) * len(y_range)))

        
    # Calculate success rate
    test_params = product(drone_max_speeds, [('Jerk_Primitive', 'NoControl')])
    for (drone_max_speed, (planner, gaze_method)) in test_params:
        axis_range = [40, 250, 460]
        for start_pos, target_pos in product(product(axis_range, axis_range), product(axis_range, axis_range)):
            if start_pos != target_pos:
                params.gaze_method = gaze_method
                params.planner = planner
                params.drone_max_speed = drone_max_speed
                params.init_position = start_pos
                params.target_list = [target_pos]
                params.record = True
                # params.render = True
                experiment = Experiment(params, result_dir)
                experiment.run()

print(all_metrics)
metric_dict = {"metric":all_metrics}
df = pd.DataFrame(metric_dict)
df.to_csv("metrics_size_validation.csv")