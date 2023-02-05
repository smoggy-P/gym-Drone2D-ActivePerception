import gym
import numpy as np
import pandas as pd
# from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl
from stable_baselines3 import PPO
# import matplotlib.pyplot as plt

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford,
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
    (df.T).to_csv(dir, index=True)

class Experiment:
    def __init__(self, params, dir):
        self.params = params
        self.env = gym.make(params.env, params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params)
        self.max_step = 20000
        self.result_dir = dir
        self.model = None

        if self.params.trained_policy:
            self.model = PPO.load(path='./trained_policy/lookahead.zip')
            self.model.set_env(self.env)

        # self.model = Qnet(action_dim=self.action_space.shape[0])


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
                index = (self.params.gaze_method,
                         self.params.agent_number,
                         self.params.pillar_number,
                         self.params.drone_view_depth, 
                         self.params.drone_view_range, 
                         self.params.agent_max_speed,
                         self.params.drone_max_speed)
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
                self.policy.__init__(self.policy, self.params)
                

            rewards.append(reward)
            steps.append(i)
            if self.params.render:
                self.env.render()
        return success, fail

import easydict

# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"

if __name__ == '__main__':

    result_dir = './experiment/results_1.csv'
    img_dir = './experiment/fails/new/'

    cfg = easydict.EasyDict({
        'env':'gym-2d-perception-v2',
        'gaze_method':'Oxford',
        'trained_policy':False,
        'policy_dir':'./trained_policy/lookahead.zip',
        'render':True,
        'dt':0.1,
        'map_scale':10,
        'map_size':[640,480],
        'agent_number':5,
        'agent_max_speed':20,
        'agent_radius':10,
        'drone_max_speed':30,
        'drone_max_acceleration':20,
        'drone_radius':5,
        'drone_max_yaw_speed':80,
        'drone_view_depth' : 80,
        'drone_view_range': 90,
        'record': False,
        'record_img': False,
        'pillar_number':3,
        'img_dir':img_dir,
        'max_steps':800
    })

    gaze_methods = ['Oxford']
    agent_numbers = [10]
    drone_view_depths = [80]
    drone_view_ranges = [90]
    pillar_numbers = [5]
    agent_max_speeds = [30]
    drone_max_speeds = [20]



    for gaze_method, agent_number, drone_view_depth, drone_view_range, pillar_number, agent_speed, drone_speed in zip(gaze_methods, 
                                                                                                                      agent_numbers, 
                                                                                                                      drone_view_depths, 
                                                                                                                      drone_view_ranges, 
                                                                                                                      pillar_numbers, 
                                                                                                                      agent_max_speeds, 
                                                                                                                      drone_max_speeds):
        cfg.gaze_method = gaze_method
        cfg.agent_number = agent_number
        cfg.drone_view_depth = drone_view_depth
        cfg.drone_view_range = drone_view_range
        cfg.pillar_number = pillar_number
        cfg.agent_max_speed = agent_speed
        cfg.drone_max_speed = drone_speed
        runner = Experiment(cfg, result_dir)
        runner.run()