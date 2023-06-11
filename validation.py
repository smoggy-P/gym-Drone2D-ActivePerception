import gym
import numpy as np
import pandas as pd
import random

from yaw_planner import Oxford, LookAhead, NoControl, Rotating, Owl, LookGoal
from datetime import datetime
from utils import *
from itertools import product


result_dir = './experiment/validation/results_'+str(datetime.now())+'.csv'

def add_to_csv(dir, value):
    df = pd.read_csv(dir, index_col=False)
    df.loc[len(df)] = value
    df.to_csv(dir, index=False)

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford,
    'Rotating': Rotating,
    'Owl' : Owl,
    'LookGoal' : LookGoal
}

gaze_methods = ['LookAhead', 'Owl']
planners = ['Jerk_Primitive']


# Environment difficulty
agent_numbers = [30]
agent_sizes = [15]
map_ids = range(1)
env_params = product(agent_numbers, agent_sizes, map_ids)

# Problem difficulty
drone_max_speeds = [20, 40, 60]
test_params = product(drone_max_speeds, planners, gaze_methods)

for (agent_number, agent_size, map_id) in env_params:

    params = Params(debug=True, 
                 map_id=map_id,
                 gaze_method='NoControl',
                 planner='NoMove',
                 agent_number=agent_number,
                 agent_radius=agent_size)
    env = gym.make('gym-2d-perception-v2', params=params)
    
    # Calculate difficulty
    position_step = 60
    T = 12
    x_range = range(params.map_scale + params.drone_radius, params.map_size[0] - params.map_scale - params.drone_radius, position_step)
    y_range = range(params.map_scale + params.drone_radius, params.map_size[1] - params.map_scale - params.drone_radius, position_step)
    total_survive = 0
    env = gym.make(params.env, params=params)
    for x in x_range:
        for y in y_range:
            # Give random velocity to agents
            env.reset()
            for agent in env.agents:
                vel = random.random() * 40 + 20
                angle = random.random() * 2 * np.pi
                agent.pref_velocity = np.array([vel * np.cos(angle), vel * np.sin(angle)])
            
            for t in np.arange(0, T, 0.1):
                env.drone.x = x
                env.drone.y = y
                _, _, done, info = env.step(0)
                env.render()
                if info['collision_flag'] == 2:
                    break
            total_survive += t

    print(total_survive / (len(x_range) * len(y_range)))
    

    # Calculate success rate

    for (drone_max_speed, planner, gaze_method) in test_params:

        axis_range = [40, 250, 460]
        
        for start_pos, target_pos in product(product(axis_range, axis_range), product(axis_range, axis_range)):
            if start_pos != target_pos:
                params.gaze_method = gaze_method
                params.planner = planner
                params.drone_max_speed = drone_max_speed
                params.init_position = start_pos
                params.target_list = [target_pos]
                policy = policy_list[params.gaze_method]
                env = gym.make('gym-2d-perception-v2', params=params)

                env.reset()
                for agent in env.agents:
                    vel = random.random() * 40 + 20
                    angle = random.random() * 2 * np.pi
                    agent.pref_velocity = np.array([vel * np.cos(angle), vel * np.sin(angle)])

            
                done = False
                while not done:
                    a = policy.plan(policy, env.info)
                    _, _, done, info = env.step(a)
                    env.render()

                    if done:
                        if params.record:
                            tracking_time = np.array([len(tracker.ts)*0.1 for tracker in info['tracker_buffer']]).sum()

                            value = (params.gaze_method,
                                    params.planner,
                                    
                                    params.motion_profile,
                                    params.map_id,
                                    params.agent_radius,
                                    params.agent_number,
                                    params.pillar_number,
                                    -1, # -1 for random agent velocity
                                    params.drone_max_speed,
                                    params.var_cam,
                                    params.init_position,
                                    params.target_list[0],


                                    info['flight_time'],
                                    info['drone'].map.grid_map.shape[0] * info['drone'].map.grid_map.shape[1] - np.sum(np.where(info['drone'].map.grid_map == 0, 1, 0)),# grid discovered
                                    len(info['tracker_buffer']),
                                    tracking_time / len(info['tracker_buffer']),

                                    1 if info['state_machine'] == state_machine['GOAL_REACHED'] else 0,
                                    1 if info['collision_flag'] == 1 else 0,
                                    1 if info['collision_flag'] == 2 else 0,
                                    info['freezing_flag'],
                                    info['dead_lock_flag'],
                                    info['state_machine'])

                            add_to_csv(result_dir, value)