import gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford
}

def add_success(dir, index):
    """Add success record

    Args:
        index = ('Oxford',1,10)
    """

    df = pd.read_csv(dir, index_col=['Method','Map','Number of agents']).T
    
    if index in df.T.index:
        df[index]['Success'] += 1
    else:
        df[index] = {'Success': 1,'Static Collision':0,'Dynamic Collision': 0}
    (df.T).to_csv("./experiment/results.csv", index=True)

def add_static_collision(dir, index):
    """Add record

    Args:
        index = ('Oxford',1,10)
    """

    df = pd.read_csv(dir, index_col=['Method','Map','Number of agents']).T
    
    if index in df.T.index:
        df[index]['Static Collision'] += 1
    else:
        df[index] = {'Success': 0,'Static Collision':1,'Dynamic Collision': 0}
    (df.T).to_csv("./experiment/results.csv", index=True)

def add_dynamic_collision(dir, index):
    """Add record

    Args:
        index = ('Oxford',1,10)
    """

    df = pd.read_csv(dir, index_col=['Method','Map','Number of agents']).T
    
    if index in df.T.index:
        df[index]['Dynamic Collision'] += 1
    else:
        df[index] = {'Success': 0,'Static Collision':0,'Dynamic Collision': 1}
    (df.T).to_csv("./experiment/results.csv", index=True)

class Experiment:
    def __init__(self, params, dir):
        self.params = params
        self.env = gym.make('gym-2d-perception-v0', params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params)
        self.max_step = 50000
        self.action_space = np.arange(-self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed/3)
        self.result_dir = dir
        # self.model = Qnet(action_dim=self.action_space.shape[0])


    def run(self):
        rewards = []
        steps = []
        success = 0
        fail = 0
        self.env.reset()
        for i in tqdm(range(self.max_step)):
            a = self.policy.plan(self.policy, self.env.info)
            state, reward, done, info = self.env.step(a)
            if info['state_machine'] == 1:
                add_success(self.result_dir,(self.params.gaze_method,1,self.params.agent_number))
            if info['collision_flag'] == 1:
                add_static_collision(self.result_dir,(self.params.gaze_method,1,self.params.agent_number))
            elif info['collision_flag'] == 2:
                add_dynamic_collision(self.result_dir,(self.params.gaze_method,1,self.params.agent_number))

            if done:
                self.env.reset()

            rewards.append(reward)
            steps.append(i)
            if self.params.render:
                self.env.render()
        return success, fail