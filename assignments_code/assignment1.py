from typing import Tuple, List, Dict
from environments.environment_abstract import Environment, State


def value_iteration_step(env: Environment, states: List[State], state_vals: Dict[State, float],
                         discount: float) -> Tuple[float, Dict[State, float]]:
    change: float = 0  # DUMMY VALUE

    return change, state_vals


def get_action(env: Environment, state: State, state_vals: Dict[State, float], discount: float) -> int:
    action: int = 0  # DUMMY VALUE
    return action
