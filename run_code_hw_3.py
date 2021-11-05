from typing import List
from environments.environment_abstract import State, Environment
from utils import env_utils
from torch import nn

from argparse import ArgumentParser
from collections import OrderedDict
import re

from code_hw.code_hw3 import astar
import numpy as np
import torch

import time
import pickle


class FullyConnectedModel(nn.Module):
    def __init__(self, input_dim: int, layer_dims: List[int], layer_acts: List[str]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        self.flatten_nn = nn.Flatten()

        # layers
        for layer_dim, act in zip(layer_dims, layer_acts):
            module_list = nn.ModuleList()

            # linear
            module_list.append(nn.Linear(input_dim, layer_dim))
            # module_list[-1].bias.data.zero_()

            # activation
            if act.upper() == "RELU":
                module_list.append(nn.ReLU())
            elif act.upper() != "LINEAR":
                raise ValueError(f"Unknown activation function {act}")

            self.layers.append(module_list)

            input_dim = layer_dim

    def forward(self, x):
        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


def load_nnet(model_file: str) -> nn.Module:
    nnet: nn.Module = FullyConnectedModel(81, [100, 100, 100, 1], ["relu", "relu", "relu", "linear"])

    # get state dict
    state_dict = torch.load(model_file)

    # remove module prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = re.sub('^module\.', '', k)
        new_state_dict[k] = v

    # set state dict
    nnet.load_state_dict(new_state_dict)

    nnet.eval()

    return nnet


def load_data():
    data_file: str = "data/puzzle8.pkl"
    data_dict = pickle.load(open(data_file, "rb"))

    states = data_dict['states']
    ctgs = data_dict['output'][:, 0]

    states_ret: List[State] = []
    ctgs_ret: List[float] = []

    num_per_ctg: int = 10
    for ctg in np.sort(np.unique(ctgs)):
        ctg_idxs = np.where(ctgs == ctg)[0][0:num_per_ctg]

        states_i = [states[idx] for idx in ctg_idxs]
        states_ret.extend(states_i)
        ctgs_ret.extend([ctg] * len(states_i))

    return states_ret, ctgs_ret


def heuristic_fn(value_net, states: List[State], env: Environment):
    nnet_inputs_np = env.states_to_nnet_input(states)
    nnet_inputs = torch.tensor(nnet_inputs_np)
    state_vals: np.array = value_net(nnet_inputs.float()).cpu().data.numpy()[:, 0]

    return -state_vals


def is_valid_soln(state: State, soln: List[int], env: Environment) -> bool:
    for move in soln:
        state, _ = env.sample_transition(state, move)

    return env.is_terminal(state)


def get_soln_cost(state: State, soln: List[int], env: Environment) -> float:
    cost: float = 0
    for move in soln:
        state, rw = env.sample_transition(state, move)

        transition_cost: float = -rw
        cost += transition_cost

    return cost


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, default="puzzle8", help="")
    parser.add_argument('--weight_g', type=float, default=1.0, help="")
    parser.add_argument('--weight_h', type=float, default=1.0, help="")

    args = parser.parse_args()

    # get environment
    env, viz, states = env_utils.get_environment(args.env)
    torch.set_num_threads(1)

    # get heuristic fn
    value_net = load_nnet("nnets/value_net.pt")
    value_net.eval()

    def heuristic_fn_help(states_inp):
        return heuristic_fn(value_net, states_inp, env)

    # load data
    states, ctgs_shortest = load_data()
    num_exs: int = len(states)

    costs: List[float] = []
    start_time = time.time()
    for state_idx, state in enumerate(states):
        start_time_i = time.time()

        soln = astar(state, env, heuristic_fn_help, args.weight_g, args.weight_h)
        assert is_valid_soln(state, soln, env), "solution must be valid"

        cost = get_soln_cost(state, soln, env)
        costs.append(cost)

        print("%i/%i - cost: %i, shortest path cost: %i, "
              "time: %.2f" % (state_idx+1, num_exs, cost, ctgs_shortest[state_idx], time.time() - start_time_i))

    print("Avg cost: %.2f, Avg shortest path cost: %.2f "
          "Total time: %s" % (float(np.mean(costs)), float(np.mean(ctgs_shortest)), time.time() - start_time))


if __name__ == "__main__":
    main()
