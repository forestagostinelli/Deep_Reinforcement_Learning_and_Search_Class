from typing import List
from environments.environment_abstract import Environment, State

import numpy as np

import torch


class ReplayBuffer:
    def __init__(self, env: Environment, max_size: int):
        self.env = env
        self.replay_buff: List = []
        self.max_size: int = max_size

    def add_experience(self, state: State, action: int, reward: float, state_next):
        if len(self.replay_buff) >= self.max_size:
            self.replay_buff.pop(0)

        self.replay_buff.append((state, action, reward, state_next))

    def samp_data(self, batch_size: int):
        batch_size_i: int = min(len(self.replay_buff), batch_size)
        batch_idxs = np.random.randint(len(self.replay_buff), size=batch_size_i)
        replay_buff_batch = [self.replay_buff[idx] for idx in batch_idxs]

        states_nnet = torch.tensor(self.env.states_to_nnet_input([x[0] for x in replay_buff_batch])).float()
        actions = torch.unsqueeze(torch.tensor(np.array([x[1] for x in replay_buff_batch])), 1)
        rewards = torch.tensor(np.array([x[2] for x in replay_buff_batch])).float()
        states_next = [x[3] for x in replay_buff_batch]

        states_next_nnet = torch.tensor(self.env.states_to_nnet_input(states_next)).float()
        is_terminal = torch.tensor(np.array([self.env.is_terminal(state_next) for state_next in states_next]))

        return states_nnet, actions, rewards, states_next_nnet, is_terminal

    def size(self):
        return len(self.replay_buff)
