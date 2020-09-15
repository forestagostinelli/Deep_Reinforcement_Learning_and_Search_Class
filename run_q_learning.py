from typing import Tuple
import numpy as np
from environments.farm_grid_world import FarmGridWorld, mask_to_idxs
from visualizer.farm_visualizer import InteractiveFarm

from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")
    parser.add_argument('--rand_right', type=float, default=0.0, help="")
    parser.add_argument('--learning_rate', type=float, default=0.5, help="")
    parser.add_argument('--epsilon', type=float, default=0.1, help="")
    parser.add_argument('--wait_greedy', type=float, default=0.1, help="")
    parser.add_argument('--wait_step', type=float, default=0.0, help="")

    args = parser.parse_args()

    grid = np.loadtxt(args.map)
    grid = np.transpose(grid)

    assert np.sum(grid == 1) == 1, "Only one agent allowed"
    assert np.sum(grid == 2) == 1, "Only one goal allowed"
    agent_idx: Tuple[int, int] = mask_to_idxs(grid, 1)[0]

    env: FarmGridWorld = FarmGridWorld(grid, args.rand_right)

    viz: InteractiveFarm = InteractiveFarm(env, agent_idx, args.discount, "ACTION", show_policy=False,
                                           wait=args.wait_greedy, val_min=-30)
    viz.q_learning(args.epsilon, args.learning_rate, args.wait_step)

    viz.mainloop()


if __name__ == "__main__":
    main()
