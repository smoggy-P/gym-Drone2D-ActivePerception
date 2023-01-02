from param.arg_utils import get_args
from experiment import Experiment

import gym
from stable_baselines3 import A2C
from stable_baselines3.common.env_checker import check_env
# from param.arg_utils import get_args

if __name__ == '__main__':
    cfg = get_args()
    runner = Experiment(cfg)
    runner.run()

# if __name__ == '__main__':

#     params = get_args()
#     env = gym.make('gym-2d-perception-v0', params=params)  # continuous: LunarLanderContinuous-v2
#     check_env(env)
#     env.is_render = True
#     model = A2C("MlpPolicy", env, verbose=1)
#     model.learn(total_timesteps=10_000)

#     vec_env = model.get_env()
#     vec_env.is_render = True
#     obs = vec_env.reset()

#     for i in range(1000):
#         action, _state = model.predict(obs, deterministic=True)

#         print(action)

#         obs, reward, done, info = vec_env.step(action)
#         vec_env.render()