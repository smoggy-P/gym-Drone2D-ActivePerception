#!/usr/bin/env python3
"""Train a PPO gaze policy with Stable Baselines 3.

Example
-------
::

    python script/train.py
"""

import os

import gym
import torch as th
from stable_baselines3 import PPO

import drone2d.envs  # noqa: F401  — register environments
from drone2d.config import SimConfig
from model.extractor import ImgStateExtractor

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def main() -> None:
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    config = SimConfig(
        env="drone-2d-perception-v2",
        gaze_method="LookAhead",
        render=False,
        record=False,
        dt=0.1,
        map_scale=10,
        map_size=[640, 480],
        agent_number=8,
        agent_max_speed=30,
        agent_radius=10,
        drone_max_speed=20,
        drone_max_acceleration=20,
        drone_radius=5,
        drone_max_yaw_speed=80,
        drone_view_depth=80,
        drone_view_range=90,
        pillar_number=5,
        max_flight_time=80,
    )

    env = gym.make(config.env, config=config)

    policy_kwargs = dict(
        net_arch=[512, dict(pi=[256], vf=[256])],
        normalize_images=False,
        features_extractor_class=ImgStateExtractor,
        features_extractor_kwargs=dict(
            cnn_encoder_name="CnnEncoder",
            device=device,
        ),
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log="./experiment/log/",
        policy_kwargs=policy_kwargs,
    )

    print("Start training")
    model.learn(total_timesteps=1_000_000)
    model.save("./trained_policy/policy_v2.zip")
    print("Model saved to ./trained_policy/policy_v2.zip")


if __name__ == "__main__":
    main()
