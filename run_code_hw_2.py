from typing import List, Dict
from environments.environment_abstract import State, Environment
from utils import env_utils
from torch import nn

from argparse import ArgumentParser

from code_hw.code_hw2 import supervised, deep_q_learning, get_value_net, deep_vi, follow_greedy_policy
import numpy as np
import torch

import time
import pickle


def get_action_val_diff(dqn: nn.Module, env: Environment, states: List[State], action_vals: Dict[State, List[float]]):
    dqn.eval()
    states_np = env.states_to_nnet_input(states)
    action_values_dqn_np = dqn(torch.tensor(states_np.astype(np.float32))).cpu().data.numpy()

    diffs_l = []
    for state_idx, state in enumerate(states):
        diffs_state = list(np.abs(np.array(action_vals[state]) - action_values_dqn_np[state_idx]))
        diffs_l.extend(diffs_state)

    print("Action mean absolute error: Mean/Min/Max (Std): %.2f/%.2f/%.2f "
          "(%.2f)" % (float(np.mean(diffs_l)), np.min(diffs_l), np.max(diffs_l),
                      float(np.std(diffs_l))))


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--algorithm', type=str, required=True, help="supervised, value_iteration, q_learning")
    parser.add_argument('--epsilon', type=float, default=0.5, help="epsilon-greedy policy")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")
    parser.add_argument('--grade', default=False, action='store_true', help="")

    args = parser.parse_args()

    # get environment
    env, viz, states = env_utils.get_environment(args.env)
    torch.set_num_threads(1)

    if args.algorithm == "supervised":
        start_time = time.time()
        data_file: str = "data/puzzle8.pkl"
        data_dict = pickle.load(open(data_file, "rb"))

        states_nnet = env.states_to_nnet_input(data_dict['states'])
        values_gt = -data_dict['output']

        value_net: nn.Module = get_value_net()
        supervised(value_net, states_nnet, values_gt, 100, 10000)

        # test
        value_net.eval()

        out_nnet = value_net(torch.tensor(states_nnet)).cpu().data.numpy()
        mse_total: float = float(np.mean((out_nnet - values_gt) ** 2))

        print("Final MSE: %f" % mse_total)
        print("Time: %f (secs)" % (time.time() - start_time))

    elif args.algorithm == "value_iteration":
        start_time = time.time()

        value_net: nn.Module = get_value_net()
        deep_vi(value_net, env, 20000, 50, 200, 100)

        # test
        num_test: int = 1000
        states_test: List[State] = env.sample_start_states(num_test)
        num_solved: int = 0
        for state in states_test:
            state_end = follow_greedy_policy(value_net, env, state, 30)
            if env.is_terminal(state_end):
                num_solved += 1

        print("Solved: %i/%i" % (num_solved, num_test))
        print("Time: %f (secs)" % (time.time() - start_time))
    elif args.algorithm == "q_learning":
        start_time = time.time()
        dqn: nn.Module = deep_q_learning(env, args.epsilon, args.discount, 1000, 30, 100, 10000, 20, viz)

        """
        # test
        data_file: str = "data/action_vals_aifarm_0.pkl"
        action_vals: Dict[State, List[float]] = pickle.load(open(data_file, "rb"))

        get_action_val_diff(dqn, env, states, action_vals)
        """
        print("Time: %f (secs)" % (time.time() - start_time))
    else:
        raise ValueError("Unknown algorithm %s" % args.algorithm)

    print("DONE")

    if viz is not None:
        viz.mainloop()


if __name__ == "__main__":
    main()
