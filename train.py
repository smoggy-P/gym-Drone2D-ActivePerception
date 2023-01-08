from param.arg_utils import get_args
from experiment import Experiment
import easydict
import gym
import pygame
import torch as th
from stable_baselines3 import A2C,PPO
from stable_baselines3.common.env_checker import check_env
from model.extractor import StackedImgStateExtractor, ImgStateExtractor

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

params = easydict.EasyDict({
        'env':'gym-2d-perception-v1',
        'gaze_method':'LookAhead',
        'render':True,
        'dt':0.1,
        'map_scale':10,
        'map_size':[640,480],
        'agent_number':10,
        'agent_max_speed':20,
        'agent_radius':10,
        'drone_max_speed':40,
        'drone_max_acceleration':15,
        'drone_radius':5,
        'drone_max_yaw_speed':80,
        'drone_view_depth' : 80,
        'drone_view_range': 120,
        'record': False,
        'pillar_number':3
    })

alg_params = {
    "policy_kwargs": dict(
        net_arch=[512, dict(pi=[256], vf=[256])],
        normalize_images=False,
        features_extractor_class=ImgStateExtractor,
        features_extractor_kwargs=dict(
            device=device, cnn_encoder_name="CnnEncoder"
        ),
    ),
    "learning_rate": 1e-5,
    "gamma": 0.99,
    "n_steps": 128,
    "batch_size": 512,
    "n_epochs": 5,
    "clip_range": 0.2,
    "ent_coef": 0.025,
    "vf_coef": 0.5,
    "target_kl": 0.01,
}

env = gym.make('gym-2d-perception-v1', params=params) 
check_env(env)
model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            device=device,
            **alg_params
        )
model.learn(total_timesteps=10000)
pygame.display.quit()
vec_env = model.get_env()
vec_env.is_render = True
obs = vec_env.reset()