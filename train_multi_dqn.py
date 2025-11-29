
import time
start_time = time.time()

import sys
import gym
import json
import random
import argparse
import numpy as np

from gym import spaces
from spatialmath import base
from datetime import datetime
from collections import namedtuple, deque
from roboticstoolbox.backends.PyPlot import PyPlot

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import spatialmath as sm
import spatialgeometry as sg
import roboticstoolbox as rtb

from libraries import utils
from libraries import crustcrawler

from torch.utils.tensorboard import SummaryWriter

# **********************************************************************

# python train_multi_dqn_pos_exp1.py --file_name multi_dqn_pos_exp1 --learning_rate 0.0025 --gamma 0.90 --batch_size 256

parser = argparse.ArgumentParser(description='Configurações dos parâmetros do treinamento.')
parser.add_argument('--file_name', type=str, help='Nome do experimento.', default = "file_name")
parser.add_argument('--learning_rate', type=float, help='Taxa de aprendizagem.', default = 0.001)
parser.add_argument('--gamma', type=float, help='Fator de desconto.', default = 0.90)
parser.add_argument('--batch_size', type=int, help='Tamanho do batch.', default = 64)

args = parser.parse_args()

EXP_NAME = args.file_name

BATCH_SIZE = args.batch_size
GAMMA = args.gamma
LR = args.learning_rate
TAU = 0.005
MEMORY = 1_000_000

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.97

epsilon = EPS_DECAY

NUM_EPISODES = 3_000
NUM_STEPS = 300

NUM_NEURONS = 64


# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

writer = SummaryWriter(f"./tensorboard/train/{EXP_NAME}")


# **********************************************************************

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
# **********************************************************************

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, NUM_NEURONS, dtype=torch.float64)
        self.layer2 = nn.Linear(NUM_NEURONS, NUM_NEURONS, dtype=torch.float64)
        self.layer3 = nn.Linear(NUM_NEURONS, n_actions, dtype=torch.float64)
        self.initialize_weights()
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(torch.float64)  # Certifique-se de que x esteja em float64
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

    def initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
    
# **********************************************************************

def select_action_1(state):
    global epsilon
    if np.random.random() > epsilon:
        with torch.no_grad():
            return policy_net_1(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def select_action_2(state):
    global epsilon
    if np.random.random() > epsilon:
        with torch.no_grad():
            return policy_net_2(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def select_action_3(state):
    global epsilon
    if np.random.random() > epsilon:
        with torch.no_grad():
            return policy_net_3(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def select_action_4(state):
    global epsilon
    if np.random.random() > epsilon:
        with torch.no_grad():
            return policy_net_4(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# **********************************************************************

def convert_time(seconds): 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

# **********************************************************************

def execution_date():
    now = datetime.now() 
    return now.strftime("dt_%d-%m-%Y_hr_%H-%M-%S")

# **********************************************************************

env = crustcrawler.CrustCrawlerEnv()

n_actions = env.action_space.n

n_observations = env.observation_space.shape[0]

policy_net_1 = DQN(n_observations, n_actions).to(device)
target_net_1 = DQN(n_observations, n_actions).to(device)
target_net_1.load_state_dict(policy_net_1.state_dict())

policy_net_2 = DQN(n_observations, n_actions).to(device)
target_net_2 = DQN(n_observations, n_actions).to(device)
target_net_2.load_state_dict(policy_net_2.state_dict())

policy_net_3 = DQN(n_observations, n_actions).to(device)
target_net_3 = DQN(n_observations, n_actions).to(device)
target_net_3.load_state_dict(policy_net_3.state_dict())

policy_net_4 = DQN(n_observations, n_actions).to(device)
target_net_4 = DQN(n_observations, n_actions).to(device)
target_net_4.load_state_dict(policy_net_4.state_dict())


optimizer_1 = optim.AdamW(policy_net_1.parameters(), lr=LR, amsgrad=True)
optimizer_2 = optim.AdamW(policy_net_2.parameters(), lr=LR, amsgrad=True)
optimizer_3 = optim.AdamW(policy_net_3.parameters(), lr=LR, amsgrad=True)
optimizer_4 = optim.AdamW(policy_net_4.parameters(), lr=LR, amsgrad=True)


memory_1 = ReplayMemory(MEMORY)
memory_2 = ReplayMemory(MEMORY)
memory_3 = ReplayMemory(MEMORY)
memory_4 = ReplayMemory(MEMORY)

total_reward = []
average_list = []
target_achieved_list = []
distance_list = []
orientation_list = []
number_steps_list = []
epsilon_list = []


# **********************************************************************

def optimize_model_1(episode, step):
    if len(memory_1) < BATCH_SIZE:
        return
    
    transitions_1 = memory_1.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions_1
    # to Transition of batch-arrays.
    batch_1 = Transition(*zip(*transitions_1))

    # Compute a mask of non-final states and concatenate the batch_1 elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch_1.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch_1.next_state
                                                if s is not None])
    state_batch_1 = torch.cat(batch_1.state)
    action_batch_1 = torch.cat(batch_1.action)
    reward_batch_1 = torch.cat(batch_1.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch_1 state according to policy_net
    state_action_values_1 = policy_net_1(state_batch_1).gather(1, action_batch_1)

    # print(state_action_values_1)
    # print()
    # sys.exit(0)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values_1 = torch.zeros(BATCH_SIZE, dtype=torch.float64, device=device)
    with torch.no_grad():
        next_state_values_1[non_final_mask] = target_net_1(non_final_next_states.to(torch.float64)).max(1).values
    # Compute the expected Q values
    expected_state_action_values_1 = (next_state_values_1 * GAMMA) + reward_batch_1

    # Compute Huber loss
    criterion_1 = nn.SmoothL1Loss()
    loss_1 = criterion_1(state_action_values_1, expected_state_action_values_1.unsqueeze(1))

    # Optimize the model
    optimizer_1.zero_grad()
    loss_1.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_1.parameters(), 100)
    optimizer_1.step()

    writer.add_scalar('Train/a) Loss', loss_1.item(), episode + step)

# ---------------------------------------------------------------------------------

def optimize_model_2(episode, step):
    if len(memory_2) < BATCH_SIZE:
        return
    
    transitions_2 = memory_2.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions_2
    # to Transition of batch-arrays.
    batch_2 = Transition(*zip(*transitions_2))

    # Compute a mask of non-final states and concatenate the batch_2 elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch_2.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch_2.next_state
                                                if s is not None])
    state_batch_2 = torch.cat(batch_2.state)
    action_batch_2 = torch.cat(batch_2.action)
    reward_batch_2 = torch.cat(batch_2.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch_2 state according to policy_net
    state_action_values_2 = policy_net_2(state_batch_2).gather(1, action_batch_2)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values_2 = torch.zeros(BATCH_SIZE, dtype=torch.float64, device=device)
    with torch.no_grad():
        next_state_values_2[non_final_mask] = target_net_2(non_final_next_states.to(torch.float64)).max(1).values
    # Compute the expected Q values
    expected_state_action_values_2 = (next_state_values_2 * GAMMA) + reward_batch_2

    # Compute Huber loss
    criterion_2 = nn.SmoothL1Loss()
    loss_2 = criterion_2(state_action_values_2, expected_state_action_values_2.unsqueeze(1))

    # Optimize the model
    optimizer_2.zero_grad()
    loss_2.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_2.parameters(), 100)
    optimizer_2.step()

# ---------------------------------------------------------------------------------

def optimize_model_3(episode, step):
    if len(memory_3) < BATCH_SIZE:
        return
    
    transitions_3 = memory_3.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions_3
    # to Transition of batch-arrays.
    batch_3 = Transition(*zip(*transitions_3))

    # Compute a mask of non-final states and concatenate the batch_3 elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch_3.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch_3.next_state
                                                if s is not None])
    state_batch_3 = torch.cat(batch_3.state)
    action_batch_3 = torch.cat(batch_3.action)
    reward_batch_3 = torch.cat(batch_3.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch_3 state according to policy_net
    state_action_values_3 = policy_net_3(state_batch_3).gather(1, action_batch_3)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values_3 = torch.zeros(BATCH_SIZE, dtype=torch.float64, device=device)
    with torch.no_grad():
        next_state_values_3[non_final_mask] = target_net_3(non_final_next_states.to(torch.float64)).max(1).values
    # Compute the expected Q values
    expected_state_action_values_3 = (next_state_values_3 * GAMMA) + reward_batch_3

    # Compute Huber loss
    criterion_3 = nn.SmoothL1Loss()
    loss_3 = criterion_3(state_action_values_3, expected_state_action_values_3.unsqueeze(1))

    # Optimize the model
    optimizer_3.zero_grad()
    loss_3.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_3.parameters(), 100)
    optimizer_3.step()

# ---------------------------------------------------------------------------------

def optimize_model_4(episode, step):
    if len(memory_4) < BATCH_SIZE:
        return
    
    transitions_4 = memory_4.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions_4
    # to Transition of batch-arrays.
    batch_4 = Transition(*zip(*transitions_4))

    # Compute a mask of non-final states and concatenate the batch_4 elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch_4.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch_4.next_state
                                                if s is not None])
    state_batch_4 = torch.cat(batch_4.state)
    action_batch_4 = torch.cat(batch_4.action)
    reward_batch_4 = torch.cat(batch_4.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch_4 state according to policy_net
    state_action_values_4 = policy_net_4(state_batch_4).gather(1, action_batch_4)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values_4 = torch.zeros(BATCH_SIZE, dtype=torch.float64, device=device)
    with torch.no_grad():
        next_state_values_4[non_final_mask] = target_net_4(non_final_next_states.to(torch.float64)).max(1).values
    # Compute the expected Q values
    expected_state_action_values_4 = (next_state_values_4 * GAMMA) + reward_batch_4

    # Compute Huber loss
    criterion_4 = nn.SmoothL1Loss()
    loss_4 = criterion_4(state_action_values_4, expected_state_action_values_4.unsqueeze(1))

    # Optimize the model
    optimizer_4.zero_grad()
    loss_4.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net_4.parameters(), 100)
    optimizer_4.step()


# **********************************************************************

for i_episode in range(1, NUM_EPISODES+1):
    rewards_list = []
    
    # Initialize the environment and get its state
    state, info = env.reset()    
    state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
    
    for t in range(1, NUM_STEPS+1):
        joint_action_1 = select_action_1(state)
        joint_action_2 = select_action_2(state)
        joint_action_3 = select_action_3(state)
        joint_action_4 = select_action_4(state)
    
        actions = [joint_action_1, joint_action_2, 
                   joint_action_3, joint_action_4]
        
        observation, reward, terminated, _ = env.step(actions)
        rewards_list.append(reward)
        
        reward = torch.tensor([reward], device=device)
        done = terminated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float64, device=device).unsqueeze(0)

        # Store the transition in memory
        memory_1.push(state, joint_action_1, next_state, reward)
        memory_2.push(state, joint_action_2, next_state, reward)
        memory_3.push(state, joint_action_3, next_state, reward)
        memory_4.push(state, joint_action_4, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model_1(i_episode, t)
        optimize_model_2(i_episode, t)
        optimize_model_3(i_episode, t)
        optimize_model_4(i_episode, t)

        # Soft update of the target network's weights
        target_net_1_state_dict = target_net_1.state_dict()
        policy_net_1_state_dict = policy_net_1.state_dict()
        for key in policy_net_1_state_dict:
            target_net_1_state_dict[key] = policy_net_1_state_dict[key]*TAU + target_net_1_state_dict[key]*(1-TAU)
        target_net_1.load_state_dict(target_net_1_state_dict)

        target_net_2_state_dict = target_net_2.state_dict()
        policy_net_2_state_dict = policy_net_2.state_dict()
        for key in policy_net_2_state_dict:
            target_net_2_state_dict[key] = policy_net_2_state_dict[key]*TAU + target_net_2_state_dict[key]*(1-TAU)
        target_net_2.load_state_dict(target_net_2_state_dict)

        target_net_3_state_dict = target_net_3.state_dict()
        policy_net_3_state_dict = policy_net_3.state_dict()
        for key in policy_net_3_state_dict:
            target_net_3_state_dict[key] = policy_net_3_state_dict[key]*TAU + target_net_3_state_dict[key]*(1-TAU)
        target_net_3.load_state_dict(target_net_3_state_dict)

        target_net_4_state_dict = target_net_4.state_dict()
        policy_net_4_state_dict = policy_net_4.state_dict()
        for key in policy_net_4_state_dict:
            target_net_4_state_dict[key] = policy_net_4_state_dict[key]*TAU + target_net_4_state_dict[key]*(1-TAU)
        target_net_4.load_state_dict(target_net_4_state_dict)
        
        if done: break

    total_reward.append(np.sum(rewards_list))
    average_list.append(np.sum(total_reward)/i_episode)
    target_achieved_list.append(1 if env.target_achieved else 0)
    distance_list.append(min(env.distances_list))
    orientation_list.append(min(env.orientations_list))
    number_steps_list.append(t)
    epsilon_list.append(epsilon)


    # Registrar a recompensa no TensorBoard
    writer.add_scalar('Info/a) Target Achieved', target_achieved_list[-1], i_episode)
    writer.add_scalar('Info/c) Number of Steps', number_steps_list[-1], i_episode)
    writer.add_scalar('Info/f) Epsilon', epsilon_list[-1], i_episode)

    writer.add_scalar('Pose Error/a) Position Error', distance_list[-1], i_episode)
    writer.add_scalar('Pose Error/b) Orientation Error', orientation_list[-1], i_episode)
    
    writer.add_scalar('Reward/a) Reward', total_reward[-1], i_episode)
    writer.add_scalar('Reward/b) Average', average_list[-1], i_episode)

    epsilon *= EPS_DECAY
    if epsilon < EPS_END:
        epsilon = EPS_END

    print(f"Episode: {i_episode} de {NUM_EPISODES} ===== Reward: {total_reward[-1]:.4f} ===== " \
          f"Avarege: {average_list[-1]:.4f} ===== Epsilon: {epsilon_list[-1]:.4f}")


print(f'Complete {EXP_NAME} training!')

