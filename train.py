from param.arg_utils import get_args
from experiment import Experiment
import easydict
import gym
import pygame
import torch as th
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_checker import check_env
from model.extractor import ImgStateExtractor
import os

if __name__ == '__main__':

    os.environ['CUDA_LAUNCH_BLOCKING']='1'
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    params = easydict.EasyDict({
            'env':'gym-2d-perception-v2',
            'gaze_method':'LookAhead',
            'render':False,
            'dt':0.1,
            'map_scale':10,
            'map_size':[640,480],
            'agent_number':10,
            'agent_max_speed':30,
            'agent_radius':10,
            'drone_max_speed':20,
            'drone_max_acceleration':20,
            'drone_radius':5,
            'drone_max_yaw_speed':80,
            'drone_view_depth' : 80,
            'drone_view_range': 90,
            'record': False,
            'record_img':False,
            'pillar_number':5,
            'max_steps':800
        })

    env = gym.make('gym-2d-perception-v1', params=params) 
    alg_params = {
        "policy_kwargs": dict(
            net_arch=[512, dict(pi=[256], vf=[256])],
            normalize_images=False,
            features_extractor_class=ImgStateExtractor,
            features_extractor_kwargs=dict(
                cnn_encoder_name="CnnEncoder",
                device=device
            ),
        ),
    }

    model = PPO(
                "MultiInputPolicy",
                env,
                verbose=1,
                device=device,
                tensorboard_log='./experiment/log/',
                **alg_params
            )


    # model = PPO.load(path='./trained_policy/lookahead.zip')
    # model.set_env(env)
    print("Start training")
    model.learn(total_timesteps=400000)
    model.save('./trained_policy/lookahead_fixed_episode.zip')
    print("successfully write model")