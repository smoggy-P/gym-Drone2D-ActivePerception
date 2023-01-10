import gym
import numpy as np
import pandas as pd
# from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl
# import matplotlib.pyplot as plt

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford
}

def add_to_csv(dir, index, flag):
    df = pd.read_csv(dir, index_col=['Method','Number of agents','Number of pillars', 'View depth', 'View range', 'Agent speed', 'Drone speed']).T
    
    if index in df.T.index:
        df[index][flag] += 1
    else:
        if flag == 'Success':
            df[index] = [1,0,0]
        elif flag == 'Static Collision':
            df[index] = [0,1,0]
        elif flag == 'Dynamic Collision':
            df[index] = [0,0,1]
    (df.T).to_csv("dir", index=True)

# def add_success(dir, index):
#     """Add success record

#     Args:
#         index = ('Oxford',1,10)
#     """

#     df = pd.read_csv(dir, index_col=['Method','Number of agents','Number of pillars', 'View depth', 'View range']).T
    
#     if index in df.T.index:
#         df[index]['Success'] += 1
#     else:
#         df[index] = [1,0,0]
#     (df.T).to_csv("./experiment/results.csv", index=True)

# def add_static_collision(dir, index):
#     """Add record

#     Args:
#         index = ('Oxford',1,10)
#     """

#     df = pd.read_csv(dir, index_col=['Method','Number of agents','Number of pillars', 'View depth', 'View range']).T
    
#     if index in df.T.index:
#         df[index]['Static Collision'] += 1
#     else:
#         df[index] = [0,1,0]
#     (df.T).to_csv("./experiment/results.csv", index=True)

# def add_dynamic_collision(dir, index):
#     """Add record

#     Args:
#         index = ('Oxford',1,10)
#     """

#     df = pd.read_csv(dir, index_col=['Method','Number of agents','Number of pillars', 'View depth', 'View range']).T
    
#     if index in df.T.index:
#         df[index]['Dynamic Collision'] += 1
#     else:
#         df[index] = [0,0,1]
#     (df.T).to_csv(dir, index=True)

class Experiment:
    def __init__(self, params, dir):
        self.params = params
        self.env = gym.make(params.env, params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params)
        self.max_step = 10000
        self.result_dir = dir
        # self.model = Qnet(action_dim=self.action_space.shape[0])


    def run(self):
        rewards = []
        steps = []
        success = 0
        fail = 0
        self.env.reset()
        for i in range(self.max_step):
            a = self.policy.plan(self.policy, self.env.info)
            state, reward, done, info = self.env.step(a)

            # print(state['local_map'].shape == state['swep_map'].shape)

            if self.params.record:
                index = (self.params.gaze_method,self.params.agent_number,self.params.pillar_number,self.params.drone_view_depth, self.params.drone_view_range)
                if info['state_machine'] == 1:
                    add_to_csv(self.result_dir,index,'Success')
                    # add_success(self.result_dir,index)
                if info['collision_flag'] == 1:
                    add_to_csv(self.result_dir,index,'Static Collision')
                    # add_static_collision(self.result_dir,index)
                elif info['collision_flag'] == 2:
                    add_to_csv(self.result_dir,index,'Dynamic Collision')
                    # add_dynamic_collision(self.result_dir,index)

            if done:
                self.env.reset()

            rewards.append(reward)
            steps.append(i)
            if self.params.render:
                self.env.render()
        return success, fail