from typing import List, Tuple, Dict, Optional
import numpy as np
import tkinter
from tkinter import Button, Frame, Label, Canvas
from tkinter import LEFT, RIGHT

from environments.farm_grid_world import FarmState, FarmGridWorld

from PIL import ImageTk, Image

import time
import os
import pickle

from assignments_code_answers.assignment1 import value_iteration_step, get_action
from assignments_code_answers.assignment2 import policy_evaluation_step, q_learning_step


def hsl_interp(frac):
    # Given frac a float in 0...1, return a color in a pleasant red-to-green
    # color scale with HSL interpolation.
    #
    # This implementation is directly drawn from
    # https://github.com/HeRCLab/nocsim
    #
    # Rather than transliterating into Python, we simply embed the TCL source
    # code, as this application already depends on TCL for Tk anyway.
    #
    # The interpolated color is returned in #RRGGBB hex format.

    tcl_src = """
# utility methods for interacting with colors in nocviz

# https://github.com/gka/chroma.js/blob/master/src/io/hsl/rgb2hsl.js

proc fmod {val mod} {
 set res $val
 while {$res > $mod} {
  set res [expr $res - $mod]
 }
 return $res
}

proc rgb2hsl {r g b} {
 set rp [expr (1.0 * $r) / 255.0]
 set gp [expr (1.0 * $g) / 255.0]
 set bp [expr (1.0 * $b) / 255.0]
 set max [::tcl::mathfunc::max $rp $gp $bp]
 set min [::tcl::mathfunc::min $rp $gp $bp]
 set delta [expr $max - $min]

 set h 0

 if {$delta == 0} {
  set h 0
 } elseif {$max == $rp} {
  set h [fmod [expr ($gp - $bp) / $delta] 6]
  set h [expr 60 * $h]
 } elseif {$max == $gp} {
  set h [expr 60 *  (( ($bp - $rp) / $delta ) + 2) ]
 } elseif {$max == $bp} {
  set h [expr 60 *  (( ($rp - $gp) / $delta ) + 4) ]
 }

 set l [expr ($max + $min) / 2.0]

 set s 0
 if {$delta == 0} {
  set s 0
 } else {
  set s [expr $delta / (1.0 - abs(2.0 * $l - 1.0)) ]
 }

 return [list [expr $h] [expr 100 * $s] [expr 100 * $l]]
}

proc hsl2rgb {h s l} {

 set s [expr (1.0 * $s) / 100.0]
 set l [expr (1.0 * $l) / 100.0]

 set c [expr (1.0 - abs(2.0 * $l - 1.0)) * $s]
 set i [fmod [expr ((1.0 * $h) / 60)] 2]
 set x [expr $c * (1.0 - abs($i - 1.0))]
 set m [expr $l - $c / 2.0]

 set rp 0
 set gp 0
 set bp 0

 if {$h < 60} {
  set rp $c
  set gp $x
  set bp 0
 } elseif {$h < 120} {
  set rp $x
  set gp $c
  set bp 0
 } elseif {$h < 180} {
  set rp 0
  set gp $c
  set bp $x
 } elseif {$h < 240} {
  set rp 0
  set gp $x
  set bp $c
 } elseif {$h < 300} {
  set rp $x
  set gp 0
  set bp $c
 } elseif {$h < 360} {
  set rp $c
  set gp 0
  set bp $x
 }

 return [list [expr int(($rp + $m) * 255)] [expr int(($gp + $m) * 255)] [expr int(($bp + $m) * 255)]]

}

# HSL interpolate drawn from https://github.com/jackparmer/colorlover

proc interp_linear {frac start end} {
 return [expr $start + ($end - $start) * $frac]
}

proc interp_circular {frac start end} {
 set s_mod [fmod $start 360]
 set e_mod [fmod $end 360]
 if { [expr max($s_mod, $e_mod) - min($s_mod, $e_mod)] > 180 } {
  if {$s_mod < $e_mod} {
   set s_mod [expr $s_mod + 360]
  } else {
   set e_mod [expr $e_mod + 360]
  }
  return [fmod [interp_linear $frac $s_mod $e_mod] 360]
 } else {
  return [interp_linear $frac $s_mod $e_mod]
 }
}

# interpolate between two HSL color tuples
#
# frac should be in 0..1
proc hsl_interp {frac h1 s1 l1 h2 s2 l2} {
 return [list [interp_circular $frac $h1 $h2] [interp_circular $frac $s1 $s2] [interp_circular $frac $l1 $l2]]

}

# interpolate between RGB colors using HSL
proc rgb_interp {frac r1 g1 b1 r2 g2 b2} {
 set hsl1 [rgb2hsl $r1 $g1 $b1]
 set hsl2 [rgb2hsl $r2 $g2 $b2]
 return [hsl2rgb {*}[hsl_interp $frac {*}$hsl1 {*}$hsl2]]
}
"""

    # clamp frac
    epsilon = 0.00001
    frac = min(max(frac, 0 + epsilon), 1.0 - epsilon)

    # grab a TCL interpreter for us to run our code in
    r=tkinter.Tcl()
    r.eval(tcl_src)

    color1 = (165, 0, 38)
    color2 = (0, 104, 55)

    # FFI into TCL to make the function call
    result = r.eval("rgb_interp {} {} {} {} {} {} {}".format(frac, *color1, *color2))

    # put it into CSS-style format
    return "#{:02x}{:02x}{:02x}".format(*[abs(int(float(s))) & 0xff for s in result.split()])


def _get_color(val: float, cell_score_min: float, cell_score_max: float):
    green_dec = int(min(255.0, max(0.0, (val - cell_score_min) * 255.0 / (cell_score_max - cell_score_min))))
    red = hex(255 - green_dec)[2:]
    green = hex(green_dec)[2:]
    if len(green) == 1:
        green += "0"
    if len(red) == 1:
        red += "0"
    color = "#" + red + green + "00"

    return color


class InteractiveFarm:
    def __init__(self, env: FarmGridWorld, start_idx: Tuple[int, int], discount: float, val_type: str,
                 show_policy: bool = True, wait: float = 0.1, val_min: Optional[float] = None):
        # 0: up, 1: down, 2: left, 3: right

        super().__init__()
        # initialize environment
        self.start_idx = start_idx
        self.agent_idx = start_idx

        self.wait: float = wait
        self.val_type: str = val_type.upper()
        self.show_policy: bool = show_policy
        self.val_min: float = val_min
        self.val_max: float = 0

        self.env: FarmGridWorld = env
        self.discount: float = discount

        self.num_actions: int = 4
        self.states: List[FarmState] = env.enumerate_states()
        self.state_vals: Dict[FarmState, float] = dict()
        self.action_vals: Dict[FarmState, List[float]] = dict()
        self.state_visits: Dict[FarmState, int] = dict()
        for state in self.states:
            self.state_visits[state] = 0
            self.state_vals[state] = 0.0

            if self.env.is_terminal(state):
                self.action_vals[state] = [0] * self.num_actions
            else:
                self.action_vals[state] = [0] * self.num_actions

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
        self._place_imgs(self.board, self.rocks_pic, env.rocks_idxs)
        self.agent_img = self._place_imgs(self.board, self.robot_pic, [self.agent_idx])[0]

        # create grid arrows
        self.grid_arrows: List[List[List]] = []

        if self.val_type == "STATE":
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

        if self.val_type == "ACTION":
            self._init_action_vals_color()

        # self.update()

        # self.monte_carlo_policy_evaluation()
        # self.td_policy_evaluation(5)
        # self.td_lambda_policy_evaluation(0.5)
        # self.policy_evaluation()
        # self.q_learning()

        self.window.update()

    def save_board(self, *_):
        print("SAVED")
        self.board.postscript(file="screenshot.eps")
        img = Image.open("screenshot.eps")
        img.save("screenshot.png", "png")

    def policy_evaluation(self, num_eval_itrs: int, policy: Dict[FarmState, List[float]], wait: float):
        if num_eval_itrs == -1:
            num_eval_itrs = np.inf

        change: float = np.inf
        itr: int = 0
        while (change > 0) and (itr < num_eval_itrs):
            # policy evaluation step
            change, self.state_vals = policy_evaluation_step(self.env, self.states, self.state_vals, policy,
                                                             self.discount)
            itr += 1

            # visualize
            if wait > 0.0:
                time.sleep(wait)
                self._update_state_vals_color()
                self._update_state_vals_text()
                self.window.update()
            print("Policy evaluation itr: %i, Delta: %E" % (itr, change))

    def policy_iteration(self, num_eval_itrs: int, wait_eval: float):
        policy: Dict[FarmState, List[float]] = {}
        for state in self.states:
            policy[state] = [0.25, 0.25, 0.25, 0.25]

        def _update():
            self._update_state_vals_color()
            self._update_state_vals_text()
            self._update_policy(policy)
            self.window.update()

        _update()

        policy_changed: bool = True
        itr: int = 0
        while policy_changed:
            # policy evaluation
            self.policy_evaluation(num_eval_itrs, policy, wait_eval)

            # policy improvement
            policy_new: Dict[FarmState, List[float]] = {}
            for state in self.states:
                action: int = get_action(self.env, state, self.state_vals, self.discount)
                policy_new[state] = [0.0, 0.0, 0.0, 0.0]
                policy_new[state][action] = 1.0

            # check for convergence
            policy_changed = policy != policy_new
            policy = policy_new
            itr += 1

            # visualize
            if self.wait > 0.0:
                _update()
                time.sleep(self.wait)

            print("Policy iteration itr: %i" % itr)

        _update()

        print("DONE")

    def value_iteration(self):
        policy: Dict[FarmState, List[float]] = {}
        for state in self.states:
            policy[state] = [1.0, 0.0, 0.0, 0.0]

        def _update():
            self._update_state_vals_color()
            self._update_state_vals_text()
            self._update_policy(policy)
            self.window.update()

        _update()

        change: float = np.inf
        itr: int = 0

        while change > 0:
            change, self.state_vals = value_iteration_step(self.env, self.states, self.state_vals, self.discount)
            itr += 1

            for state in self.states:
                action: int = get_action(self.env, state, self.state_vals, self.discount)
                policy[state] = [0.0, 0.0, 0.0, 0.0]
                policy[state][action] = 1.0

            if self.wait > 0.0:
                time.sleep(self.wait)
                _update()

            print("VI Itr: %i, Delta: %E" % (itr, change))

        _update()

        # save answer
        actions: List[int] = []
        for state in self.states:
            action: int = get_action(self.env, state, self.state_vals, self.discount)
            actions.append(action)

        # pickle.dump((self.state_vals, actions, itr), open("value_iteration.pkl", "wb"), protocol=-1)

        print("DONE")

    def q_learning(self, epsilon: float, learning_rate: float, wait_step: float):
        state: FarmState = FarmState(self.start_idx)

        def _update():
            self._update_action_vals_color()
            self.window.update()

        episode_num: int = 0
        print("Q-learning, episode %i" % episode_num)
        while episode_num < 1000:
            if self.env.is_terminal(state):
                episode_num = episode_num + 1
                if episode_num % 100 == 0:
                    print("Visualizing greedy policy")
                    _update()
                    self.greedy_policy_vis(40)
                state = FarmState(self.start_idx)

                print("Q-learning, episode %i" % episode_num)

            state, self.action_vals = q_learning_step(self.env, state, self.action_vals, epsilon, learning_rate,
                                                      self.discount)

            if wait_step > 0.0:
                self.board.delete(self.agent_img)
                self.agent_img = self._place_imgs(self.board, self.robot_pic, [state.idx])[0]

                _update()
                time.sleep(wait_step)

        _update()

        print("DONE")

    def greedy_policy_vis(self, num_steps: int):
        def _update():
            self.window.update()

        curr_state = FarmState(self.start_idx)

        self.board.delete(self.agent_img)
        self.agent_img = self._place_imgs(self.board, self.robot_pic, [curr_state.idx])[0]
        _update()
        time.sleep(self.wait)

        print("Step: ", end='', flush=True)
        for itr in range(num_steps):
            print("%i..." % itr, end='', flush=True)

            if self.env.is_terminal(curr_state):
                break

            action: int = int(np.argmax(self.action_vals[curr_state]))
            curr_state, _ = self.env.sample_transition(curr_state, action)

            self.board.delete(self.agent_img)
            self.agent_img = self._place_imgs(self.board, self.robot_pic, [curr_state.idx])[0]

            _update()
            time.sleep(self.wait)

        print("")

    def mainloop(self):
        self.window.mainloop()

    def _update_state_vals_text(self):
        for state in self.states:
            pos_i, pos_j = state.idx
            val: float = self.state_vals[state]

            self.board.itemconfigure(self.grid_text[pos_i][pos_j], text=str(format(val, '.2f')), fill="black")

    def _update_state_vals_color(self):
        cell_score_max: float = self.val_max
        if self.val_min is None:
            cell_score_min: float = min(self.state_vals.values()) - 1E-9
        else:
            cell_score_min: float = self.val_min

        for state in self.states:
            pos_i, pos_j = state.idx
            val: float = self.state_vals[state]

            # color = _get_color(val, cell_score_min, cell_score_max)
            color = hsl_interp((val - cell_score_min) / (cell_score_max - cell_score_min))

            self.board.itemconfigure(self.grid_squares[pos_i][pos_j], fill=color)

            self.board.itemconfigure(self.grid_text[pos_i][pos_j], text=str(format(val, '.2f')), fill="black")

    def _init_action_vals_color(self):
        grid_dim_x, grid_dim_y = self.env.grid_shape

        self.action_val_arrows = []
        for pos_i in range(grid_dim_x):
            grid_arrows_row: List = []
            for pos_j in range(grid_dim_y):

                state_action_val_arrows: List = []
                state = FarmState((pos_i, pos_j))
                if not self.env.is_terminal(state):
                    for action in range(self.num_actions):
                        grid_arrow = self._create_arrow(action, pos_i, pos_j, "white")
                        state_action_val_arrows.append(grid_arrow)

                grid_arrows_row.append(state_action_val_arrows)

            self.action_val_arrows.append(grid_arrows_row)

    def _update_action_vals_color(self):
        cell_score_max: float = self.val_max
        if self.val_min is None:
            cell_score_min: float = min(self.state_vals.values()) - 1E-9
        else:
            cell_score_min: float = self.val_min

        grid_dim_x, grid_dim_y = self.env.grid_shape

        for pos_i in range(grid_dim_x):
            for pos_j in range(grid_dim_y):
                state: FarmState = FarmState((pos_i, pos_j))
                if self.env.is_terminal(state):
                    continue

                for action in range(self.num_actions):
                    action_val: float = self.action_vals[state][action]
                    color = _get_color(action_val, cell_score_min, cell_score_max)
                    self.board.itemconfigure(self.action_val_arrows[pos_i][pos_j][action], fill=color)

    def _update_policy(self, policy: Dict[FarmState, List[float]]):
        grid_dim_x, grid_dim_y = self.env.grid_shape

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

                for action, policy_prob in enumerate(policy[state]):
                    if policy_prob == 0.0:
                        continue
                    color: str = "gray%i" % (100 - 100 * policy_prob)
                    grid_arrow = self._create_arrow(action, pos_i, pos_j, color)
                    grid_arrows_row.append(grid_arrow)

            self.grid_arrows.append(grid_arrows_row)

    def _create_arrow(self, action: int, pos_i: int, pos_j: int, color):
        triangle_size: float = 0.2

        if action == 0:
            grid_arrow = self.board.create_polygon((pos_i + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_j + triangle_size) * self.width + 4,
                                                   (pos_i + 0.5 + triangle_size) * self.width + 4,
                                                   (pos_j + triangle_size) * self.width + 4,
                                                   (pos_i + 0.5) * self.width + 4,
                                                   pos_j * self.width + 4,
                                                   fill=color, width=1)
        elif action == 1:
            grid_arrow = self.board.create_polygon((pos_i + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_j + 1 - triangle_size) * self.width + 4,
                                                   (pos_i + 0.5 + triangle_size) * self.width + 4,
                                                   (pos_j + 1 - triangle_size) * self.width + 4,
                                                   (pos_i + 0.5) * self.width + 4,
                                                   (pos_j + 1) * self.width + 4,
                                                   fill=color, width=1)

        elif action == 2:
            grid_arrow = self.board.create_polygon((pos_i + triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_i + triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 + triangle_size) * self.width + 4,
                                                   pos_i * self.width + 4,
                                                   (pos_j + 0.5) * self.width + 4,
                                                   fill=color, width=1)
        elif action == 3:
            grid_arrow = self.board.create_polygon((pos_i + 1 - triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 - triangle_size) * self.width + 4,
                                                   (pos_i + 1 - triangle_size) * self.width + 4,
                                                   (pos_j + 0.5 + triangle_size) * self.width + 4,
                                                   (pos_i + 1) * self.width + 4,
                                                   (pos_j + 0.5) * self.width + 4,
                                                   fill=color, width=1)
        else:
            raise ValueError("Unknown action %i" % action)

        return grid_arrow

    def _place_imgs(self, board: Canvas, img, idxs: List[Tuple[int, int]]):
        created_imgs: List = []
        for idx in idxs:
            created_img = board.create_image(idx[0] * self.width + self.width_half + 4,
                                             idx[1] * self.width + self.width_half + 4, image=img)
            created_imgs.append(created_img)

        return created_imgs
