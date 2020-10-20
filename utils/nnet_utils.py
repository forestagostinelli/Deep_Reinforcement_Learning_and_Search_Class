import torch

import torch.optim as optim
from torch.optim.optimizer import Optimizer
import torch.nn as nn

import numpy as np


def train_state_value_step(nnet: nn.Module, nnet_inputs_np: np.ndarray, nnet_targets_np: np.ndarray,
                           device: torch.device, train_itr: int, lr: float, lr_d: float) -> float:
    # initialize
    nnet.train()

    criterion = nn.MSELoss()  # loss function, mean squared error
    optimizer: Optimizer = optim.Adam(nnet.parameters(), lr=lr)

    # zero the parameter gradients
    optimizer.zero_grad()

    # set learning rate
    lr_itr = lr * (lr_d ** train_itr)  # exponential decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_itr

    # send data to device (i.e. CPU or GPU)
    nnet_inputs_np = nnet_inputs_np.astype(np.float32)
    nnet_targets_np = nnet_targets_np.astype(np.float32)

    nnet_inputs = torch.tensor(nnet_inputs_np, device=device)
    nnet_targets = torch.tensor(nnet_targets_np, device=device)

    # forward
    nnet_outputs = nnet(nnet_inputs)

    # loss
    loss = criterion(nnet_outputs, nnet_targets)

    # backpropagation
    loss.backward()

    # step
    optimizer.step()

    return loss.item()
