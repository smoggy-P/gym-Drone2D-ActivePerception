import gym
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl
from model.DQN import preprocess, Qnet

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl,
    'Oxford': Oxford
}

class Experiment:
    def __init__(self, params, dir):
        self.params = params
        self.env = gym.make('gym-2d-perception-v0', params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params)
        self.max_step = 50000
        self.action_space = np.arange(-self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed, self.params.drone_max_yaw_speed/3)
        self.df = pd.read_csv("./experiment/results.csv")
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
            
            if reward == 100:
                success += 1
            if reward == -100:
                fail += 1
            if done:
                self.env.reset()

            rewards.append(reward)
            steps.append(i)
            if self.params.render:
                self.env.render()
        return success, fail