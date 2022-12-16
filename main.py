from param.arg_utils import get_args
from experiment import Experiment
from model.DQN import DQN, train_DQN
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import rl_utils
import random

if __name__ == '__main__':
    cfg = get_args()
    runner = Experiment(cfg)
    runner.run()