from typing import List, Dict, Optional, Tuple

from utils.rl_utils import ReplayBuffer

from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmState

import torch
import numpy as np
from visualizer.farm_visualizer import InteractiveFarm

from utils import misc_utils

from torch import nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
import time


def update_dqn_action_vals(env: Environment, viz: InteractiveFarm, dqn):
    dqn.eval()

    states: List[FarmState] = viz.states
    states_np = env.states_to_nnet_input(states)
    action_values_np = dqn(torch.tensor(states_np.astype(np.float32))).cpu().data.numpy()

    action_values: Dict[FarmState, List[float]] = dict()
    for state_idx, state in enumerate(states):
        action_values[state] = list(action_values_np[state_idx])

    viz.set_action_values(action_values)


def update_state(viz: InteractiveFarm, state_curr: State):
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state_curr.agent_idx])[0]

    viz.window.update()


def expand(env: Environment, states: List[State]) -> Tuple[List[List[State]], np.ndarray]:
    """

    @param env: the environment
    @param states: a list of states

    @return: a list of expanded states (list of lists of states), a 2D numpy that represents the rewards when
    transitioning to each state
    """
    num_actions: int = len(env.get_actions())

    states_expanded: List[List[State]] = []
    rewards_expanded: np.ndarray = np.zeros((len(states), num_actions))
    for state_idx, state in enumerate(states):
        state_expanded: List[State] = []
        for action_idx, action in enumerate(env.get_actions()):
            state_next, reward = env.sample_transition(state, action)

            state_expanded.append(state_next)
            rewards_expanded[state_idx, action_idx] = reward

        states_expanded.append(state_expanded)

    return states_expanded, rewards_expanded


def run_greedy_dqn_policy(env: Environment, viz: InteractiveFarm, dqn):
    wait_time: float = 0.01
    dqn.eval()
    update_dqn_action_vals(env, viz, dqn)

    state = env.sample_start_states(1)[0]

    update_state(viz, state)
    time.sleep(wait_time)

    episode_step: int = 0
    while (episode_step < 30) and (not env.is_terminal(state)):
        state_np = env.states_to_nnet_input([state])

        action_values_state = dqn(torch.tensor(state_np)).cpu().data.numpy()[0, :]
        action: int = int(np.argmax(action_values_state))

        state_next, _ = env.sample_transition(state, action)

        state = state_next

        episode_step += 1

        update_state(viz, state)
        time.sleep(wait_time)


def get_value_net() -> nn.Module:
    """

    @return: neural network for the value function
    """
    pass


def supervised(value_net: nn.Module, states_np: np.ndarray, values_np: np.ndarray, batch_size: int, num_itrs: int):
    """

    @param value_net: PyTorch state value neural network
    @param states_np: numpy representation of the states
    @param values_np: the optimal values
    @param batch_size: the batch size for training
    @param num_itrs: the number of training iterations
    """
    pass


def follow_greedy_policy(value_net: nn.Module, env: Environment, state: State, num_steps: int) -> State:
    """

    @param value_net: PyTorch state value neural network
    @param env: Environment
    @param state: the starting state
    @param num_steps: the number of states to take

    @return: the state resulting from following the greedy policy
    """
    pass


def deep_vi(value_net: nn.Module, env: Environment, vi_batch_size: int, num_vi_itrs: int, nnet_train_itrs: int,
            batch_size: int):
    """

    @param value_net: PyTorch state value neural network
    @param env: Environment
    @param vi_batch_size: the number of states to generate for each iteration of value iteration
    @param num_vi_itrs: number of value iterations
    @param nnet_train_itrs: the number of iterations for which to train the neural network on each iteration of
    value iteration
    @param batch_size: the batch size when training the neural network
    """
    pass


def deep_q_learning(env: Environment, epsilon: float, discount: float, num_episodes: int, max_episode_steps: int,
                    batch_size: int, replay_buff_size: int, targ_update_episodes: int,
                    viz: Optional[InteractiveFarm]) -> nn.Module:
    """
    @param env: environment
    @param epsilon: epsilon-greedy policy
    @param discount: the discount factor
    @param num_episodes: number of episodes
    @param max_episode_steps: maximum number of steps to take in an episode
    @param batch_size: size of the training batch
    @param replay_buff_size: the size of the replay buffer
    @param targ_update_episodes: after how many episodes to update the parameters of the target dqn to the current dqn
    @param viz: optional visualizer

    @return: the action value function found by Q-learning
    """
    pass
