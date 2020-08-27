from typing import List, Tuple, Dict
import numpy as np
import tkinter
from tkinter import Button, Frame, Label, Canvas
from tkinter import LEFT, RIGHT

from environments.farm_grid_world import FarmState, FarmGridWorld

from PIL import ImageTk, Image

import time
import os

from assignments_code.assignment1 import value_iteration_step, get_action


class InteractiveFarm:
    def __init__(self, env: FarmGridWorld, start_idx: Tuple[int, int], discount: float, no_text: bool,
                 no_val: bool, no_policy: bool, wait: float):
        # 0: up, 1: down, 2: left, 3: right

        super().__init__()
        # initialize environment
        self.wait: float = wait
        self.no_text: bool = no_text
        self.no_val: bool = no_val
        self.no_policy: bool = no_policy

        self.env: FarmGridWorld = env
        self.discount: float = discount

        self.states: List[FarmState] = env.enumerate_states()
        self.state_vals: Dict[FarmState, float] = dict()
        for state in self.states:
            self.state_vals[state] = 0.0

        # initialize board
        self.window = tkinter.Tk()
        self.window.wm_title("CSE 790 Farm")

        self.width: int = 70
        self.width_half: int = int(self.width / 2)

        # load pictures
        path = os.getcwd() + "/images/"
        self.goal_pic = ImageTk.PhotoImage(file=path + 'goal.png')
        self.plant_pic = ImageTk.PhotoImage(file=path + 'plant.png')
        self.robot_pic = ImageTk.PhotoImage(file=path + 'robot.png')
        self.rocks_pic = ImageTk.PhotoImage(file=path + 'rocks.png')

        grid_dim_x, grid_dim_y = env.grid_shape

        self.board: Canvas = Canvas(self.window, width=grid_dim_y * self.width + 2, height=grid_dim_x * self.width + 2)

        # create initial grid squares
        self.grid_squares: List[List] = []
        for pos_i in range(grid_dim_x):
            grid_squares_row: List = []
            for pos_j in range(grid_dim_y):
                square = self.board.create_rectangle(pos_i * self.width + 4,
                                                     pos_j * self.width + 4,
                                                     (pos_i + 1) * self.width + 4,
                                                     (pos_j + 1) * self.width + 4, fill="white", width=1)

                grid_squares_row.append(square)
            self.grid_squares.append(grid_squares_row)

        # create figures
        self._place_imgs(self.board, self.goal_pic, [env.goal_idx])
        self._place_imgs(self.board, self.plant_pic, env.plant_idxs)
        self._place_imgs(self.board, self.robot_pic, [start_idx])
        self._place_imgs(self.board, self.rocks_pic, env.rocks_idxs)

        # create grid arrows
        self.grid_arrows: List[List[List]] = []

        if not self.no_text:
            # create initial grid values
            self.grid_text: List[List] = []
            for pos_i in range(grid_dim_x):
                grid_text_rows: List = []
                for pos_j in range(grid_dim_y):
                    val = self.board.create_text(pos_i * self.width + self.width_half,
                                                 pos_j * self.width + self.width_half,
                                                 text="", fill="black")
                    grid_text_rows.append(val)
                self.grid_text.append(grid_text_rows)

        self.board.pack(side=LEFT)

        do_buttons: bool = False
        if do_buttons:
            # make control buttons
            panel = Frame(self.window)
            panel.pack(side=RIGHT)
            Label(text="Buttons\n", font="Verdana 12 bold").pack()

            value_itr_frame = Frame(self.window)
            value_itr_frame.pack()

            b1 = Button(text="Save Figure")
            b1.bind("<Button-1>", self.save_board)
            b1.pack()

            vi_button = Button(text="Value Iteration")
            vi_button.bind("<Button-1>", self.value_iteration)
            vi_button.pack()

        self.update()

        self.value_iteration()

    def update(self):
        if not self.no_val:
            self._update_state_vals()
        if not self.no_policy:
            self._update_policy()

        self.window.update()

    def save_board(self, *_):
        print("SAVED")
        self.board.postscript(file="screenshot.eps")
        img = Image.open("screenshot.eps")
        img.save("screenshot.png", "png")

    def value_iteration(self, *_):
        change: float = np.inf
        itr: int = 0
        while change > 0:
            change, self.state_vals = value_iteration_step(self.env, self.states, self.state_vals, self.discount)
            itr += 1

            if self.wait > 0.0:
                time.sleep(0.0)
                self.update()
            print("Itr: %i, Delta: %.2f" % (itr, change))

        self.update()

        print("DONE")

    def mainloop(self):
        self.window.mainloop()

    def _update_state_vals(self):
        cell_score_min: float = -30.0
        cell_score_max: float = 0.0
        for state in self.states:
            pos_i, pos_j = state.idx
            val: float = self.state_vals[state]

            green_dec = int(min(255.0, max(0.0, (val - cell_score_min) * 255.0 / (cell_score_max - cell_score_min))))
            red = hex(255 - green_dec)[2:]
            green = hex(green_dec)[2:]
            if len(green) == 1:
                green += "0"
            if len(red) == 1:
                red += "0"
            color = "#" + red + green + "00"

            self.board.itemconfigure(self.grid_squares[pos_i][pos_j], fill=color)

            if not self.no_text:
                self.board.itemconfigure(self.grid_text[pos_i][pos_j], text=str(format(val, '.2f')), fill="black")

    def _update_policy(self):
        grid_dim_x, grid_dim_y = self.env.grid_shape

        triangle_size = 0.2
        for grid_arrows_row in self.grid_arrows:
            for grid_arrow in grid_arrows_row:
                self.board.delete(grid_arrow)

        self.grid_arrows: List[List[List]] = []
        for pos_i in range(grid_dim_x):
            grid_arrows_row: List = []
            for pos_j in range(grid_dim_y):
                state: FarmState = FarmState((pos_i, pos_j))
                if self.env.is_terminal(state):
                    continue

                action: int = get_action(self.env, state, self.state_vals, self.discount)

                fill_color = "black"
                if action == 0:
                    grid_arrow = self.board.create_polygon((pos_i + 0.5 - triangle_size) * self.width + 4,
                                                           (pos_j + triangle_size) * self.width + 4,
                                                           (pos_i + 0.5 + triangle_size) * self.width + 4,
                                                           (pos_j + triangle_size) * self.width + 4,
                                                           (pos_i + 0.5) * self.width + 4,
                                                           pos_j * self.width + 4,
                                                           fill=fill_color, width=1)
                elif action == 1:
                    grid_arrow = self.board.create_polygon((pos_i + 0.5 - triangle_size) * self.width + 4,
                                                           (pos_j + 1 - triangle_size) * self.width + 4,
                                                           (pos_i + 0.5 + triangle_size) * self.width + 4,
                                                           (pos_j + 1 - triangle_size) * self.width + 4,
                                                           (pos_i + 0.5) * self.width + 4,
                                                           (pos_j + 1) * self.width + 4,
                                                           fill=fill_color, width=1)

                elif action == 2:
                    grid_arrow = self.board.create_polygon((pos_i + triangle_size) * self.width + 4,
                                                           (pos_j + 0.5 - triangle_size) * self.width + 4,
                                                           (pos_i + triangle_size) * self.width + 4,
                                                           (pos_j + 0.5 + triangle_size) * self.width + 4,
                                                           pos_i * self.width + 4,
                                                           (pos_j + 0.5) * self.width + 4,
                                                           fill=fill_color, width=1)
                elif action == 3:
                    grid_arrow = self.board.create_polygon((pos_i + 1 - triangle_size) * self.width + 4,
                                                           (pos_j + 0.5 - triangle_size) * self.width + 4,
                                                           (pos_i + 1 - triangle_size) * self.width + 4,
                                                           (pos_j + 0.5 + triangle_size) * self.width + 4,
                                                           (pos_i + 1) * self.width + 4,
                                                           (pos_j + 0.5) * self.width + 4,
                                                           fill=fill_color, width=1)
                else:
                    raise ValueError("Unknown action %i" % action)

                grid_arrows_row.append(grid_arrow)

            self.grid_arrows.append(grid_arrows_row)

    def _place_imgs(self, board: Canvas, img, idxs: List[Tuple[int, int]]):
        created_imgs: List = []
        for idx in idxs:
            created_img = board.create_image(idx[0] * self.width + self.width_half + 4,
                                             idx[1] * self.width + self.width_half + 4, image=img)
            created_imgs.append(created_img)

        return created_imgs
