from typing import List
import torch
from torch import nn
from environments.environment_abstract import Environment, State
import numpy as np


def relu_forward(inputs):

    return outputs


def relu_backward(grad, inputs):

    return inputs_grad


def linear_forward(inputs, weights, biases):

    return outputs


def linear_backward(grad, inputs, weights, biases):

    return weights_grad, biases_grad, inputs_grad


def get_dqn():

    return dqn


def deep_q_learning_step(env: Environment, state: State, dqn: nn.Module, dqn_target: nn.Module, epsilon: float,
                         discount: float, batch_size: int, optimizer, device, replay_buffer: List):
    # get action
    dqn.eval()

    # get transition

    # add to replay buffer

    # sample from replay buffer and train
    batch_idxs = np.random.randint(len(replay_buffer), size=batch_size)

    states_nnet_np = np.concatenate([env.state_to_nnet_input(replay_buffer[idx][0]) for idx in batch_idxs], axis=0)
    actions_np = np.array([replay_buffer[idx][1] for idx in batch_idxs])
    rewards_np = np.array([replay_buffer[idx][2] for idx in batch_idxs])

    states_next = [replay_buffer[idx][3] for idx in batch_idxs]
    states_next_nnet_np = np.concatenate([env.state_to_nnet_input(replay_buffer[idx][3]) for idx in batch_idxs], axis=0)
    is_terminal_np = np.array([env.is_terminal(state_next) for state_next in states_next])

    states_nnet = torch.tensor(states_nnet_np, device=device)
    actions = torch.unsqueeze(torch.tensor(actions_np, device=device), 1)
    rewards = torch.tensor(rewards_np, device=device)
    states_next_nnet = torch.tensor(states_next_nnet_np, device=device)
    is_terminal = torch.tensor(is_terminal_np, device=device)

    # train DQN
    dqn.train()
    optimizer.zero_grad()

    # compute target

    # get output of dqn

    # loss

    # backpropagation
    loss.backward()

    # optimizer step
    optimizer.step()

    return state_next, dqn, replay_buffer
