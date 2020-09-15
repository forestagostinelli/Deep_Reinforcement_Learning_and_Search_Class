from environments.environment_abstract import State, Environment
from typing import List, Tuple
import numpy as np


def mask_to_idxs(grid: np.ndarray, mask_int: int) -> List[Tuple[int, int]]:
    pos1_np, pos2_np = np.where(grid == mask_int)
    pos1: List[int] = [int(x) for x in pos1_np]
    pos2: List[int] = [int(x) for x in pos2_np]

    return list(zip(pos1, pos2))


class FarmState(State):
    def __init__(self, idx: Tuple[int, int]):
        self.idx = idx

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return self.idx == other.idx


class FarmGridWorld(Environment):
    def __init__(self, grid: np.ndarray, rand_right: float):
        super().__init__()

        assert np.sum(grid == 2) == 1, "Only one goal allowed"
        self.rand_right_prob: float = rand_right

        self.grid_shape: Tuple[int, int] = grid.shape
        self.grass_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 0)
        self.goal_idx: Tuple[int, int] = mask_to_idxs(grid, 2)[0]
        self.plant_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 3)
        self.rocks_idxs: List[Tuple[int, int]] = mask_to_idxs(grid, 4)

    def get_actions(self) -> List[int]:
        return list(range(4))

    def enumerate_states(self) -> List[FarmState]:
        states: List[FarmState] = []

        for pos_i in range(self.grid_shape[0]):
            for pos_j in range(self.grid_shape[1]):
                state: FarmState = FarmState((pos_i, pos_j))
                states.append(state)

        return states

    def is_terminal(self, state: FarmState) -> bool:
        return state.idx == self.goal_idx

    def sample_transition(self, state: FarmState, action: int) -> Tuple[FarmState, float]:
        # 0: up, 1: down, 2: left, 3: right
        idx_curr = state.idx

        if self.is_terminal(state):
            reward: float = 0.0
            state_next = FarmState(idx_curr)
        else:
            if np.random.rand(1)[0] < self.rand_right_prob:
                state_next = FarmState(self._get_next_idx(idx_curr, 3))
            else:
                state_next = FarmState(self._get_next_idx(idx_curr, action))

            if state_next.idx in self.plant_idxs:
                reward: float = -50.0
            elif state_next.idx in self.rocks_idxs:
                reward: float = -10.0
            else:
                reward: float = -1.0

        return state_next, reward

    def state_action_dynamics(self, state: FarmState, action: int) -> Tuple[float, List[FarmState], List[float]]:
        # 0: up, 1: down, 2: left, 3: right
        idx_curr = state.idx

        if self.is_terminal(state):
            expected_reward: float = 0.0
            states_next = [FarmState(idx_curr)]
            probs = [1.0]
        else:
            states_next = [FarmState(self._get_next_idx(idx_curr, action))]

            if (self.rand_right_prob > 0) and (action != 3):
                states_next.append(FarmState(self._get_next_idx(idx_curr, 3)))
                probs = [1.0 - self.rand_right_prob, self.rand_right_prob]
            else:
                probs = [1.0]

            expected_reward: float = 0
            for state_next, prob in zip(states_next, probs):
                if state_next.idx in self.plant_idxs:
                    reward: float = -50.0
                elif state_next.idx in self.rocks_idxs:
                    reward: float = -10.0
                else:
                    reward: float = -1.0

                expected_reward += prob * reward

        assert(np.sum(probs) == 1.0)

        return expected_reward, states_next, probs

    def _get_next_idx(self, idx_curr: Tuple[int, int], action: int) -> Tuple[int, int]:
        if action == 0:
            idx_next = (idx_curr[0], max(idx_curr[1] - 1, 0))
        elif action == 1:
            idx_next = (idx_curr[0], min(idx_curr[1] + 1, self.grid_shape[1] - 1))
        elif action == 2:
            idx_next = (max(idx_curr[0] - 1, 0), idx_curr[1])
        elif action == 3:
            idx_next = (min(idx_curr[0] + 1, self.grid_shape[0] - 1), idx_curr[1])
        else:
            raise ValueError("Unknown action %i" % action)

        return idx_next
