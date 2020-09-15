from typing import Tuple, List, Dict
from environments.environment_abstract import Environment, State


def policy_evaluation_step(env: Environment, states: List[State], state_vals: Dict[State, float],
                           policy: Dict[State, List[float]], discount: float) -> Tuple[float, Dict[State, float]]:
    change: float = 0.0

    return change, state_vals


def q_learning_step(env: Environment, state: State, action_vals: Dict[State, List[float]], epsilon: float,
                    learning_rate: float, discount: float):

    return state_next, action_vals
