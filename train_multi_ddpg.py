
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

# python train_multi_ddpg_pos_exp1.py --file_name multi_ddpg_pos_exp1 --lr_actor 0.0025 --lr_critic 0.0025 --tau 0.005 --noise 0.01 --gamma 0.90 --batch_size 256 --headless False

parser = argparse.ArgumentParser(description='Configurações dos parâmetros do treinamento.')
parser.add_argument('--file_name', type=str, help='Nome do experimento.', default="file_name")
parser.add_argument('--lr_actor', type=float, help='Taxa de aprendizagem do ator.', default=0.00025)
parser.add_argument('--lr_critic', type=float, help='Taxa de aprendizagem do crítico.', default=0.00025)
parser.add_argument('--tau', type=float, help='Taxa de atualização das redes.', default=0.005)
parser.add_argument('--noise', type=float, help='Taxa de ruído', default=0.025)
parser.add_argument('--gamma', type=float, help='Fator de desconto.', default=0.90)
parser.add_argument('--batch_size', type=int, help='Tamanho do batch.', default=64)
parser.add_argument('--headless', type=str, choices=['True', 'False'], default='False', help='Modo Headless.')  

args = parser.parse_args()

EXP_NAME = args.file_name

LR_ACTOR = args.lr_actor
LR_CRITIC = args.lr_critic
TAU = args.tau
NOISE = args.noise
GAMMA = args.gamma
BATCH_SIZE = args.batch_size

MEMORY = 1_000_000

NUM_EPISODES = 3_000
NUM_STEPS = 300

NUM_NEURONS = 128

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

writer = SummaryWriter(f"./tensorboard/train/{EXP_NAME}")

# **********************************************************************

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)

    def __len__(self):
        return len(self.memory)
    
# **********************************************************************

# Rede Ator (Gera ações contínuas)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))  
        return x * ACTION_LIMIT # A saída está entre [-0.5°, 0.5°] em radianos

# Rede Crítica (Aproxima Q(s, a))
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, NUM_NEURONS)
        self.fc2 = nn.Linear(NUM_NEURONS, NUM_NEURONS)
        self.fc3 = nn.Linear(NUM_NEURONS, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# **********************************************************************

class DDPGAgent:
    def __init__(self, state_dim, action_dim):
        self.tau = TAU
        self.gamma = GAMMA
        lr_actor = LR_ACTOR 
        lr_critic = LR_CRITIC

        # Redes principais
        self.actor = Actor(state_dim, action_dim).double().to(device)
        self.critic = Critic(state_dim, action_dim).double().to(device)

        # Redes-alvo
        self.actor_target = Actor(state_dim, action_dim).double().to(device)
        self.critic_target = Critic(state_dim, action_dim).double().to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Otimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Memória de replay
        self.memory = ReplayMemory(MEMORY)

    def select_action(self, state,):
        state = torch.tensor(state, dtype=torch.float64, device=device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        action += NOISE * np.random.randn(len(action))  # Adiciona ruído para exploração
        return np.clip(action, -ACTION_LIMIT, +ACTION_LIMIT)

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return

        batch = self.memory.sample()
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float64, device=device)
        actions = torch.tensor(np.array(actions), dtype=torch.float64, device=device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float64, device=device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float64, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float64, device=device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q_values = self.critic_target(next_states, next_actions)
            y = rewards + self.gamma * target_q_values * (1 - dones)

        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


# **********************************************************************

env = KinovaEnv()

# Get the number of state observations
n_observations = env.observation_space.shape[0]

# Get number of actions from gym action space
n_actions = env.action_space.shape[0]

agent_joint_0 = DDPGAgent(n_observations, n_actions)
agent_joint_1 = DDPGAgent(n_observations, n_actions)
agent_joint_2 = DDPGAgent(n_observations, n_actions)
agent_joint_3 = DDPGAgent(n_observations, n_actions)


total_reward = []
average_list = []
target_achieved_list = []
distance_list = []
orientation_list = []
number_steps_list = []
epsilon_list = []


# **********************************************************************

for i_episode in range(1, NUM_EPISODES+1):
    rewards_list = []
          
    # Initialize the environment and get its state
    state, info = env.reset()    
    
    for t in range(1, NUM_STEPS+1):
        actions = []

        action_joint_0 = agent_joint_0.select_action(state)
        action_joint_1 = agent_joint_1.select_action(state)
        action_joint_2 = agent_joint_2.select_action(state)
        action_joint_3 = agent_joint_3.select_action(state)

        actions = [
            action_joint_0,
            action_joint_1,
            action_joint_2,
            action_joint_3

        ]

        next_state, reward, done, _ = env.step(actions)
        rewards_list.append(reward)

        agent_joint_0.memory.push(state, action_joint_0, reward, next_state, done)
        agent_joint_1.memory.push(state, action_joint_1, reward, next_state, done)
        agent_joint_2.memory.push(state, action_joint_2, reward, next_state, done)
        agent_joint_3.memory.push(state, action_joint_3, reward, next_state, done)

        agent_joint_0.train()
        agent_joint_1.train()
        agent_joint_2.train()
        agent_joint_3.train()
        
        # Move to the next state
        state = next_state

        if done: break

    total_reward.append(np.sum(rewards_list))
    average_list.append(np.sum(total_reward)/i_episode)
    target_achieved_list.append(1 if env.target_achieved else 0)
    distance_list.append(min(env.distances_list))
    orientation_list.append(min(env.orientations_list))
    number_steps_list.append(t)


    # Registrar a recompensa no TensorBoard
    writer.add_scalar('Info/a) Target Achieved', target_achieved_list[-1], i_episode)
    writer.add_scalar('Info/b) Number of Steps', number_steps_list[-1], i_episode)

    writer.add_scalar('Pose Error/a) Position Error', distance_list[-1], i_episode)
    writer.add_scalar('Pose Error/b) Orientation Error', orientation_list[-1], i_episode)
    
    writer.add_scalar('Reward/a) Reward', total_reward[-1], i_episode)
    writer.add_scalar('Reward/b) Average', average_list[-1], i_episode)


    print(f"Episode: {i_episode} de {NUM_EPISODES} ===== Reward: {total_reward[-1]:.4f} ===== " \
          f"Avarege: {average_list[-1]:.4f}")
        

print(f'Complete {EXP_NAME} training!')

