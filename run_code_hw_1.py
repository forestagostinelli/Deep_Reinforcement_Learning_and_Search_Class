from typing import List, Dict
from environments.environment_abstract import State
from utils import env_utils

from argparse import ArgumentParser

from code_hw.code_hw1 import policy_iteration, value_iteration, q_learning


def run_policy_iteration(states, env, discount, viz):
    policy: Dict[State, List[float]] = {}
    state_values: Dict[State, float] = {}
    for state in states:
        policy[state] = [0.25, 0.25, 0.25, 0.25]
        state_values[state] = 0.0
    policy_iteration(env, states, state_values, policy, discount, 0.0, viz)


def run_value_iteration(states, env, discount, viz):
    policy: Dict[State, List[float]] = {}
    state_values: Dict[State, float] = {}
    for state in states:
        policy[state] = [0.25, 0.25, 0.25, 0.25]
        state_values[state] = 0.0
    value_iteration(env, states, state_values, discount, 0.0, viz)


def run_q_learning(states, env, discount, epsilon, learning_rate, viz):
    action_values: Dict[State, List[float]] = {}
    for state in states:
        action_values[state] = [0.0, 0.0, 0.0, 0.0]

    q_learning(env, action_values, epsilon, learning_rate, discount, 1000, viz)


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--env', type=str, required=True, help="")
    parser.add_argument('--algorithm', type=str, required=True, help="policy_iteration, value_iteration, "
                                                                     "q_learning")
    parser.add_argument('--epsilon', type=float, default=0.1, help="epsilon-greedy policy")
    parser.add_argument('--learning_rate', type=float, default=0.5, help="learning rate")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")

    args = parser.parse_args()

    # get environment
    env, viz, states = env_utils.get_environment(args.env)

    if args.algorithm == "policy_iteration":
        run_policy_iteration(states, env, args.discount, viz)
    elif args.algorithm == "value_iteration":
        run_value_iteration(states, env, args.discount, viz)
    elif args.algorithm == "q_learning":
        run_q_learning(states, env, args.discount, args.epsilon, args.learning_rate, viz)
    else:
        raise ValueError("Unknown algorithm %s" % args.algorithm)

    print("DONE")

    if viz is not None:
        viz.mainloop()


if __name__ == "__main__":
    main()
