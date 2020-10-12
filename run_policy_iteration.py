import numpy as np
from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm

from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--discount', type=float, default=1.0, help="Discount")
    parser.add_argument('--num_itrs', type=int, default=-1, help="Number of iterations for policy evaluation")
    parser.add_argument('--rand_right', type=float, default=0.0, help="")
    parser.add_argument('--wait', type=float, default=0.0, help="")
    parser.add_argument('--wait_eval', type=float, default=0.0, help="")

    args = parser.parse_args()

    grid = np.loadtxt(args.map)
    grid = np.transpose(grid)

    assert np.sum(grid == 1) == 1, "Only one agent allowed"
    assert np.sum(grid == 2) == 1, "Only one goal allowed"

    env: FarmGridWorld = FarmGridWorld(grid.shape, args.rand_right)

    viz: InteractiveFarm = InteractiveFarm(env, grid, args.discount, "STATE", show_policy=True,
                                           wait=args.wait, val_min=-30)
    viz.policy_iteration(args.num_itrs, wait_eval=args.wait_eval)

    # viz.mainloop()


if __name__ == "__main__":
    main()
