import sys
sys.path.append('/home/smoggy/thesis/gym-Drone2D-ActivePerception/gym_2d_perception/envs')
sys.path.append('/home/smoggy/thesis/gym-Drone2D-ActivePerception/')

import gym
from tqdm import tqdm
from yaw_planner import Oxford, LookAhead, NoControl
from arg_utils import get_args

policy_list = {
    'LookAhead': LookAhead,
    'NoControl': NoControl
}

class Experiment:
    def __init__(self, params):
        self.params = params
        self.env = gym.make('gym-2d-perception-v0', params=params)
        self.dt = params.dt
        self.policy = policy_list[params.gaze_method]
        self.policy.__init__(self.policy, params.dt)
        self.max_step = 10000


    def run(self):
        rewards = []
        steps = []
        success = 0
        fail = 0
        for i in tqdm(range(self.max_step)):
            a = self.policy.plan(self.policy, self.env.observation)
            observation, reward, done= self.env.step(a)
            if reward == 100:
                success += 1
            if reward == -100:
                fail += 1
            if done:
                self.env.reset()

            rewards.append(reward)
            steps.append(i)
            self.env.render()

cfg = get_args()
runner = Experiment(cfg)
runner.run()