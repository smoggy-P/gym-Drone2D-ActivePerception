import gym
import numpy as np
import pandas as pd
import os
# from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl, Rotating, Owl
from stable_baselines3 import PPO
from datetime import datetime
# import matplotlib.pyplot as plt

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford,
    'Rotating': Rotating,
    'Owl' : Owl
}

def add_to_csv(dir, value):
    df = pd.read_csv(dir, index_col=False)
    df.loc[len(df)] = value
    df.to_csv(dir, index=False)

def average_in_view(arr):
    ranges = []
    arr -= 1
    for a in arr:
        # Create an array that is 1 where a is 0, and pad each end with an extra 0.
        iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        # Runs start and end where absdiff is 1.
        ranges.extend(np.where(absdiff == 1)[0].tolist())
    ranges = np.asarray(ranges).reshape(-1, 2)
    return np.sum(np.diff(ranges, axis=-1)) / ranges.shape[0] if ranges.shape[0] >0 else 0 

class Experiment:
    def __init__(self, params, dir):
        self.params = params
        self.env = gym.make(params.env, params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params)
        self.max_step = 10000
        self.result_dir = dir
        self.model = None
        
        self.last_step = 0
        self.last_undiscovered_grid = np.sum(np.where(self.env.drone.map.grid_map == 0, 1, 0))
        self.last_agent_tracked = 0

        if (not os.path.isfile(dir)) and (params.record):
            d = {'Method':[],
                 'Planner':[],
                 'Number of agents':[],
                 'Number of pillars':[], 
                 'Agent speed':[], 
                 'Drone speed':[], 
                 'Steps':[],
                 'Grid discovered':[],
                 'Agent tracked':[],
                 'Agent tracked time':[],
                 'Success':[],
                 'Static Collision':[],
                 'Dynamic Collision':[]}
            df = pd.DataFrame(d)
            df.to_csv(dir, index=False)

        if self.params.trained_policy:
            self.model = PPO.load(path='./trained_policy/lookahead.zip')
            self.model.set_env(self.env)


    def run(self):
        rewards = []
        steps = []
        success = 0
        fail = 0
        state = self.env.reset()
        for i in range(self.max_step):
            if self.params.trained_policy:
                a, state_ = self.model.predict(state, deterministic=True)
            else:
                a = self.policy.plan(self.policy, self.env.info)
            state, reward, done, info = self.env.step(a)

            if self.params.record:
                if info['state_machine'] == 1 or info['collision_flag'] == 1 or info['collision_flag'] == 2:
                    value = (self.params.gaze_method,
                             self.params.planner,
                             self.params.agent_number,
                             self.params.pillar_number,
                             self.params.agent_max_speed,
                             self.params.drone_max_speed,
                             i - self.last_step + 1, # steps during this process
                             self.last_undiscovered_grid - np.sum(np.where(info['drone'].map.grid_map == 0, 1, 0)),# grid discovered
                             self.env.tracked_agent - self.last_agent_tracked,
                             average_in_view(np.array(self.env.seen_history).T),
                             1 if info['state_machine'] == 1 else 0,
                             1 if info['collision_flag'] == 1 else 0,
                             1 if info['collision_flag'] == 2 else 0)
                    self.last_step = i
                    self.last_undiscovered_grid = np.sum(np.where(info['drone'].map.grid_map == 0, 1, 0))
                    self.last_agent_tracked = self.env.tracked_agent
                    add_to_csv(self.result_dir, value)

            if done:
                self.env.reset()
                self.last_undiscovered_grid = np.sum(np.where(self.env.drone.map.grid_map == 0, 1, 0))
                self.last_agent_tracked = 0
                self.policy.__init__(self.policy, self.params)
                

            rewards.append(reward)
            steps.append(i)
            if self.params.render:
                self.env.render()
        return success, fail
