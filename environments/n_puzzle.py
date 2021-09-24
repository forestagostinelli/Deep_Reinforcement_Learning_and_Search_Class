from typing import List, Tuple, Union
import numpy as np

from .environment_abstract import Environment, State
from random import randrange


class NPuzzleState(State):
    __slots__ = ['tiles', 'hash']

    def __init__(self, tiles: np.ndarray):
        self.tiles: np.ndarray = tiles
        self.hash = None

    def __hash__(self):
        return hash(self.tiles.tostring())

    def __eq__(self, other):
        return np.array_equal(self.tiles, other.tiles)


class NPuzzle(Environment):
    moves: List[str] = ['U', 'D', 'L', 'R']
    moves_rev: List[str] = ['D', 'U', 'R', 'L']

    def __init__(self, dim: int):
        super().__init__()

        self.dim: int = dim
        if self.dim <= 15:
            self.dtype = np.uint8
        else:
            self.dtype = np.int

        # Solved state
        self.goal_tiles: np.ndarray = np.concatenate((np.arange(1, self.dim * self.dim), [0])).astype(self.dtype)

        # Next state ops
        self.swap_zero_idxs: np.ndarray = self._get_swap_zero_idxs(self.dim)

    def get_actions(self) -> List[int]:
        return list(range(4))

    def sample_transition(self, state: NPuzzleState, action: int) -> Tuple[NPuzzleState, float]:
        reward, states_next, _ = self.state_action_dynamics(state, action)

        return states_next[0], reward

    def state_action_dynamics(self, state: NPuzzleState, action: int) -> Tuple[float, List[NPuzzleState], List[float]]:
        if self.is_terminal(state):
            states_next = [NPuzzleState(state.tiles.copy())]
            expected_reward: float = 0.0
        else:
            # initialize
            state_np = np.stack([x.tiles for x in [state]], axis=0)
            state_next_np: np.ndarray = state_np.copy()

            # get zero indicies
            z_idxs: np.ndarray
            _, z_idxs = np.where(state_next_np == 0)

            # get next state
            states_next_np, _, transition_costs = self._move_np(state_np, z_idxs, action)

            # make states
            states_next: List[NPuzzleState] = [NPuzzleState(x) for x in list(states_next_np)]

            expected_reward: float = -1.0

        return expected_reward, states_next, [1.0]

    def is_terminal(self, state: NPuzzleState) -> bool:
        is_equal = np.equal(state.tiles, self.goal_tiles)

        return np.all(is_equal, axis=0)

    def states_to_nnet_input(self, states: List[NPuzzleState]) -> np.ndarray:
        states_np = np.stack([x.tiles for x in states], axis=0)
        one_hot_mat = np.eye(self.dim ** 2)

        states_nnet = one_hot_mat[states_np]
        states_nnet = states_nnet.reshape((states_nnet.shape[0], -1))

        states_nnet = states_nnet.astype(np.float32)

        return states_nnet

    def get_num_moves(self) -> int:
        return len(self.moves)

    def sample_start_states(self, num_states: int) -> List[NPuzzleState]:
        backwards_range = (0, 2 * self.dim ** 4)

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self._generate_goal_states(num_states, np_format=True)

        # Get z_idxs
        z_idxs: np.ndarray
        _, z_idxs = np.where(states_np == 0)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        while np.max(num_back_moves < scramble_nums):
            idxs: np.ndarray = np.where((num_back_moves < scramble_nums))[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], z_idxs[idxs], _ = self._move_np(states_np[idxs], z_idxs[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1

        states: List[NPuzzleState] = [NPuzzleState(x) for x in list(states_np)]

        return states

    def _generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[NPuzzleState], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_tiles.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[NPuzzleState] = [NPuzzleState(self.goal_tiles.copy()) for _ in range(num_states)]

        return solved_states

    def _get_swap_zero_idxs(self, n: int) -> np.ndarray:
        swap_zero_idxs: np.ndarray = np.zeros((n ** 2, len(NPuzzle.moves)), dtype=self.dtype)
        for moveIdx, move in enumerate(NPuzzle.moves):
            for i in range(n):
                for j in range(n):
                    z_idx = np.ravel_multi_index((i, j), (n, n))

                    state = np.ones((n, n), dtype=np.int)
                    state[i, j] = 0

                    is_eligible: bool = False
                    if move == 'U':
                        is_eligible = i < (n - 1)
                    elif move == 'D':
                        is_eligible = i > 0
                    elif move == 'L':
                        is_eligible = j < (n - 1)
                    elif move == 'R':
                        is_eligible = j > 0

                    if is_eligible:
                        swap_i: int = -1
                        swap_j: int = -1
                        if move == 'U':
                            swap_i = i + 1
                            swap_j = j
                        elif move == 'D':
                            swap_i = i - 1
                            swap_j = j
                        elif move == 'L':
                            swap_i = i
                            swap_j = j + 1
                        elif move == 'R':
                            swap_i = i
                            swap_j = j - 1

                        swap_zero_idxs[z_idx, moveIdx] = np.ravel_multi_index((swap_i, swap_j), (n, n))
                    else:
                        swap_zero_idxs[z_idx, moveIdx] = z_idx

        return swap_zero_idxs

    def _move_np(self, states_np: np.ndarray, z_idxs: np.array,
                 action: int) -> Tuple[np.ndarray, np.array, List[float]]:
        states_next_np: np.ndarray = states_np.copy()

        # get index to swap with zero
        state_idxs: np.ndarray = np.arange(0, states_next_np.shape[0])
        swap_z_idxs: np.ndarray = self.swap_zero_idxs[z_idxs, action]

        # swap zero with adjacent tile
        states_next_np[state_idxs, z_idxs] = states_np[state_idxs, swap_z_idxs]
        states_next_np[state_idxs, swap_z_idxs] = 0

        # transition costs
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, swap_z_idxs, transition_costs
