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
    lr = 1e-2
    num_episodes = 2000
    hidden_dim = 128
    gamma = 0.98
    epsilon = 0.01
    target_update = 50
    buffer_size = 5000
    minimal_size = 1000
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")
    params = get_args()
    env_name = 'gym-2d-perception-v0'
    env = gym.make('gym-2d-perception-v0', params=params)
    action_dim = 11  # 将连续动作分成11个离散动作
    cfg = get_args()
    np.seterr(divide='ignore', invalid='ignore')
    # runner = Experiment(cfg)
    # runner.run()
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = DQN(action_dim, lr, gamma, epsilon,
                target_update, device)
    return_list, max_q_value_list = train_DQN(agent, env, num_episodes,
                                            replay_buffer, minimal_size,
                                            batch_size)

    episodes_list = list(range(len(return_list)))
    mv_return = rl_utils.moving_average(return_list, 5)

    returns = np.array(mv_return)
    np.save('returns.npy', returns)
    q_values = np.array(max_q_value_list)
    np.save('q_values.npy', q_values)

    plt.plot(episodes_list, mv_return)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()

    frames_list = list(range(len(max_q_value_list)))
    plt.plot(frames_list, max_q_value_list)
    plt.axhline(0, c='orange', ls='--')
    plt.axhline(10, c='red', ls='--')
    plt.xlabel('Frames')
    plt.ylabel('Q value')
    plt.title('DQN on {}'.format(env_name))
    plt.show()