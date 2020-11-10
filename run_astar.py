import numpy as np
from environments.farm_grid_world import FarmGridWorld
from visualizer.farm_visualizer import InteractiveFarm

from argparse import ArgumentParser


def main():
    parser: ArgumentParser = ArgumentParser()
    parser.add_argument('--map', type=str, required=True, help="")
    parser.add_argument('--g_weight', type=float, default=1.0, help="")
    parser.add_argument('--h_weight', type=float, default=1.0, help="")
    parser.add_argument('--wait', type=float, default=0.1, help="")

    args = parser.parse_args()

    grid = np.loadtxt(args.map)
    grid = np.transpose(grid)

    assert np.sum(grid == 1) == 1, "Only one agent allowed"
    assert np.sum(grid == 2) == 1, "Only one goal allowed"

    env: FarmGridWorld = FarmGridWorld(grid.shape, 0.0)

    viz: InteractiveFarm = InteractiveFarm(env, grid, 1.0, "STATE", show_policy=True,
                                           wait=args.wait, val_min=-30)
    viz.astar(args.g_weight, args.h_weight)

    viz.mainloop()


if __name__ == "__main__":
    main()
