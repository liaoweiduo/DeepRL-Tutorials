# %% md

# Deep Q Networks

# %% md

## Imports

# %%
import sys

import gym
from gym import wrappers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from IPython.display import clear_output
from matplotlib import pyplot as plt

import random
from timeit import default_timer as timer
from datetime import timedelta
import math
from utils.wrappers import make_atari, wrap_deepmind, wrap_pytorch

from utils.hyperparameters import Config
from agents.BaseAgent import BaseAgent

# %% md

# Hyperparameters

# %%

config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 30000
config.epsilon_by_frame = lambda frame_idx: config.epsilon_final + (
            config.epsilon_start - config.epsilon_final) * math.exp(-1. * frame_idx / config.epsilon_decay)

# misc agent variables
config.GAMMA = 0.99
config.LR = 1e-4

# memory
config.TARGET_NET_UPDATE_FREQ = 1000
config.EXP_REPLAY_SIZE = 100000
config.BATCH_SIZE = 32

# Learning control variables
config.LEARN_START = 1000
config.MAX_FRAMES  = 15000


# %% md

## Replay Memory

# %%

class ExperienceReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# %% md

## Network Declaration

# %%

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):  # input:10*3 actions:105
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        # self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.hidden1 = nn.Linear(self.input_shape[0] , 217)
        self.hidden2 = nn.Linear(217, 211)
        self.predict = nn.Linear(211, self.num_actions)

    def forward(self, x):
        # x = x.view(x.size(0), -1)   # 拉长成向量 第一维是同时处理的数据数 不变，后面拉长
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.predict(x)

        return x

    # def feature_size(self):
    #     return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)


# %% md

## Agent

# %%
# %%

class Model(BaseAgent):
    def __init__(self, static_policy=False, env=None, config=None):
        super(Model, self).__init__()
        self.device = config.device

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START

        self.static_policy = static_policy
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n
        self.env = env

        self.declare_networks()

        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)

    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions = self.memory.sample(self.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device,
                                      dtype=torch.bool)
        try:  # sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device,
                                                 dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values

    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values = batch_vars

        # estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)

        # target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + (self.gamma * max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.huber(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        self.save_loss(loss.item())
        self.save_sigma_param_magnitudes()

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def huber(self, x):
        cond = (x.abs() < 1.0).to(torch.float)
        return 0.5 * x.pow(2) * cond + (x.abs() - 0.5) * (1 - cond)


# %% md

## Plot Results

# %%

def plot(frame_idx, rewards, losses, sigma, elapsed_time):
    clear_output(True)
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s. time: %s' % (frame_idx, np.mean(rewards[-10:]), elapsed_time))
    plt.plot(rewards)
    if losses:
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
    if sigma:
        plt.subplot(133)
        plt.title('noisy param magnitude')
        plt.plot(sigma)
    plt.show()


# %% md

## Training Loop

# %%

start=timer()

env_id = "gym_DVSinR2HCA:DVSinR2HCA-v0"
env = gym.make(env_id)
# env    = wrap_deepmind(env, frame_stack=False)
# env    = wrap_pytorch(env)
model = Model(env=env, config=config)
# model.load_w_by_name("DQN")

episode_reward = 0

observation = env.reset()
for frame_idx in range(1, config.MAX_FRAMES + 1): # config.MAX_FRAMES
    epsilon = config.epsilon_by_frame(frame_idx)

    action = model.get_action(observation, epsilon)
    prev_observation=observation
    observation, reward, done, _ = env.step(action)
    observation = None if done else observation

    model.update(prev_observation, action, reward, observation, frame_idx)
    episode_reward += reward

    if done:
        observation = env.reset()
        model.save_reward(episode_reward)
        episode_reward = 0

        if np.mean(model.rewards[-10:]) > 19:
            # plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))
            print("\rFrame {}/{}. time {}".format(frame_idx, config.MAX_FRAMES, timedelta(seconds=int(timer()-start))), end="")
            sys.stdout.flush()
            break

    if frame_idx % 10000 == 0:
        plot(frame_idx, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))

    if frame_idx % 10 == 0:
        print("\rFrame {}/{}. time {}".format(frame_idx, config.MAX_FRAMES, timedelta(seconds=int(timer()-start))), end="")
        sys.stdout.flush()

plot(config.MAX_FRAMES + 1, model.rewards, model.losses, model.sigma_parameter_mag, timedelta(seconds=int(timer()-start)))
model.save_w_by_name(name="DQN")
env.close()

# %%     plt the rewards vs episodes, ylim -1 to 1
# rewards = model.rewards.copy()
# plt.figure(figsize=(20, 5))
# plt.title('Rewards list')
# plt.plot(rewards)
# plt.xlabel('episodes')
# plt.ylabel('total rewards')
# plt.ylim(-1, 1)
# plt.xlim(0,500)
# plt.show()

#%%     get the selected vectors
# env_id = "gym_DVSinR2HCA:DVSinR2HCA-v0"
env = gym.make(env_id)
# model = Model(env=env, config=config)
# model.load_w_by_name("DQN_36_sort")
done = False
ob = env.reset()
rewards = 0
for i in range(100):
    # action = model.get_action(ob, 0)  # greedy take action
    ob_tensor = torch.tensor([ob], device=model.device, dtype=torch.float)
    q_values = model.model(ob_tensor)
    print('q_values:')
    print(q_values)
    a = q_values.max(1)[1].view(1, 1)
    action = a.item()
    print("action:", action)
    ob, reward, done, message = env.step(action)
    print(message)
    rewards += reward
    if done is True:
        break
env.render()
# env.close()

#%%
batch_state, batch_action, batch_reward, batch_next_state = zip(*model.memory.memory)
count = 0
for i in range(len(batch_reward)):
    if batch_reward[i] == -1:
        count += 1
count