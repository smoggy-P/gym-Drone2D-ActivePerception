import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(state):
    drone = state['drone']
    trajectory = state['trajectory']
    swep_map = torch.zeros([64, 48], device=device)
    for i, pos in enumerate(trajectory.positions):
        swep_map[int(pos[0]//10), int(pos[1]//10)] = i * 0.1

    drone_map = torch.from_numpy(drone.map.grid_map).to(device)

    x = torch.stack((swep_map, drone_map)).unsqueeze(0)
    # print(x.shape)
    # plt.imshow(x[1].cpu())
    # plt.show()
    # plt.pause(0.1)
    # plt.clf()
    # print(x[0].shape)
    return x


    

class Qnet(torch.nn.Module):
    def __init__(self, action_dim):
        super(Qnet, self).__init__()
        kernel_size = 5
        padding = 2
        hidden_channel = 6
        input_dim = (64, 48)
        self.conv1 = torch.nn.Conv2d(2, hidden_channel, kernel_size=kernel_size, padding=padding).to(device)
        self.mp = torch.nn.MaxPool2d(2, stride=1).to(device)
        self.relu = torch.nn.ReLU().to(device)

        hidden_dim1 = (input_dim[0]-kernel_size+2*padding)*(input_dim[1]-kernel_size+2*padding)*hidden_channel
        hidden_dim2 = 80

        self.fc1 = torch.nn.Linear(hidden_dim1, hidden_dim2).to(device)
        self.fc2 = torch.nn.Linear(hidden_dim2, action_dim).to(device)
        self.logsoftmax = torch.nn.LogSoftmax(dim=0).to(device)

    def forward(self, x):
        # print(x.shape)
        out = self.conv1(x)
        out = self.relu(self.mp(out))
        out = out.view(x.shape[0], -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        out = self.logsoftmax(out)
        # out = torch.argmax(out)
        # print(out.shape)
        return out

class DQN:
    def __init__(self,
                 action_dim,
                 learning_rate,
                 gamma,
                 epsilon,
                 target_update,
                 dqn_type='VanillaDQN'):
        self.action_dim = action_dim
        self.q_net = Qnet(self.action_dim).to(device)
        self.target_q_net = Qnet(self.action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.dqn_type = dqn_type
        self.device = device

    def take_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = state.to(self.device)
            action = self.q_net(state).argmax().item()
        return action
    
    def max_q_value(self, state):
        state = state.to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        # print("states shape:", states.shape)
        # print("actions shape:", actions.shape)
        # print(self.q_net(states).shape)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        if self.dqn_type == 'DoubleDQN': # DQN与Double DQN的区别
            max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
            max_next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else: # DQN的情况
            max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

def dis_to_con(discrete_action, env, action_dim):  # 离散动作转回连续的函数
    action_lowbound = env.action_space[0]  # 连续动作的最小值
    action_upbound = env.action_space[-1]  # 连续动作的最大值
    return action_lowbound + (discrete_action /
                              (action_dim - 1)) * (action_upbound -
                                                   action_lowbound)

def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size,
              batch_size):
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10),
                  desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(
                        state) * 0.005 + max_q_value * 0.995  # 平滑处理
                    max_q_value_list.append(max_q_value)  # 保存每个状态的最大Q值
                    action_continuous = dis_to_con(action, env,
                                                   agent.action_dim)
                    next_state, reward, done, _ = env.step(action_continuous)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(
                            batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode':
                        '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return':
                        '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list


# a = torch.randn(2, 2, 64, 48, device=device)
# model = Qnet(20)
# print(model(a).shape)