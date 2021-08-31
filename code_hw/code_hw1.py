from typing import List, Dict, Optional, Tuple
from environments.environment_abstract import Environment, State
from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm


def update_dp(viz: InteractiveFarm, state_values, policy):
    viz.set_state_values(state_values)
    viz.set_policy(policy)
    viz.window.update()


def update_model_free(viz: InteractiveFarm, state, action_values):
    viz.set_action_values(action_values)
    viz.board.delete(viz.agent_img)
    viz.agent_img = viz.place_imgs(viz.board, viz.robot_pic, [state.agent_idx])[0]
    viz.window.update()


def policy_evaluation(env: Environment, states: List[State], state_values: Dict[State, float],
                      policy: Dict[State, List[float]], discount: float, cutoff: float) -> Dict[State, float]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param cutoff: while delta > cutoff, continue policy evaluation

    @return: state values for each state when following the given policy. This must be a dictionary that maps states
    to values
    """

    pass


def policy_improvement(env: Environment, states: List[State], state_values: Dict[State, float],
                       discount: float) -> Dict[State, List[float]]:
    """ Return policy that behaves greedily with respect to value function

    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param discount: the discount factor

    @return: the policy that behaves greedily with respect to the value function. This must be a dictionary that maps
    states to list of floats. The list of floats is the probability of taking each action.
    """
    pass


def policy_iteration(env: FarmGridWorld, states: List[State], state_values: Dict[State, float],
                     policy: Dict[State, List[float]], discount: float, policy_eval_cutoff: float,
                     viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param policy: dictionary that maps states to a list of probabilities of taking each action
    @param discount: the discount factor
    @param policy_eval_cutoff: the cutoff for policy evaluation
    @param viz: optional visualizer

    @return: the state value function and policy found by policy iteration
    """
    pass


def value_iteration(env: Environment, states: List[State], state_values: Dict[State, float],
                    discount: float, cutoff: float,
                    viz: Optional[InteractiveFarm]) -> Tuple[Dict[State, float], Dict[State, List[float]]]:
    """
    @param env: environment
    @param states: all states in the state space
    @param state_values: dictionary that maps states to values
    @param discount: the discount factor
    @param cutoff: while delta > cutoff, continue value iteration
    @param viz: optional visualizer

    @return: the state value function and policy found by value iteration
    """
    pass


def q_learning(env: Environment, action_values: Dict[State, List[float]], epsilon: float, learning_rate: float,
               discount: float, num_episodes: int, viz: Optional[InteractiveFarm]) -> Dict[State, List[float]]:
    """
    @param env: environment
    @param action_values: dictionary that maps states to their action values (list of floats)
    @param epsilon: epsilon-greedy policy
    @param learning_rate: learning rate
    @param discount: the discount factor
    @param num_episodes: number of episodes for learning
    @param viz: optional visualizer

    @return: the action value function found by Q-learning
    """
    pass
