import gym
import numpy as np
import pandas as pd
import os
# from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl, Rotating, Owl, LookGoal
from datetime import datetime
from utils import *
# import matplotlib.pyplot as plt

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford,
    'Rotating': Rotating,
    'Owl' : Owl,
    'LookGoal' : LookGoal
}

def add_to_csv(dir, value):
    df = pd.read_csv(dir, index_col=False)
    df.loc[len(df)] = value
    df.to_csv(dir, index=False)

class Experiment:
    def __init__(self, params, dir):
        self.params = params
        self.env = gym.make(params.env, params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params)
        self.result_dir = dir

        if (not os.path.isfile(dir)) and (params.record):
            d = {'Method':[],
                 'Planner':[],
                 'Motion Profile':[],
                 'Map ID':[],
                 'Agent size':[],
                 'Number of agents':[],
                 'Number of pillars':[], 
                 'Agent speed':[], 
                 'Drone speed':[], 
                 'Depth variance':[],
                 'Initial position':[],
                 'Target position':[],

                 'Flight time':[],
                 'Grid discovered':[],
                 'Agent tracked':[],
                 'Agent tracked time':[],
                 'Success':[],
                 'Static Collision':[],
                 'Dynamic Collision':[],
                 'Freezing':[],
                 'Dead Lock':[],
                 'state machine':[]}
            df = pd.DataFrame(d)
            df.to_csv(dir, index=False)


    def run(self):
        self.env.reset()
        done = False
        while not done:
            a = self.policy.plan(self.policy, self.env.info)
            _, _, done, info = self.env.step(a)

            if done:
                if self.params.record:
                    tracking_time = np.array([len(tracker.ts)*0.1 for tracker in info['tracker_buffer']]).sum()

                    value = (self.params.gaze_method,
                             self.params.planner,
                             
                             self.params.motion_profile,
                             self.params.map_id,
                             self.params.agent_radius,
                             self.params.agent_number,
                             self.params.pillar_number,
                             self.params.agent_max_speed,
                             self.params.drone_max_speed,
                             self.params.var_cam,
                             self.params.init_position,
                             self.params.target_list[0],


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

                    add_to_csv(self.result_dir, value)
                    
            if self.params.render:
                self.env.render()
