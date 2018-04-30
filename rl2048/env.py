import math
import os

import numpy as np

from . import config
from .game import Game2048


def process_state(state):

    if np.max(state) < 2:
        raise ValueError(f"Invalid state with max. value{np.max(state)}")

    state = np.array(state, dtype=np.float, copy=True)
    positive_mask = state > 0
    positive_states = state[positive_mask]

    state[positive_mask] = np.log2(positive_states)/11.0
    state[np.logical_not(positive_mask)] = -1.0

    return state


class Env2048(object):

    def __init__(self, step_limit=np.inf):

        conf = config.get_config()
        self.game = Game2048()
        self.action_map = conf.action_map
        self.empty_state = np.zeros((4, 4), dtype=np.int)
        self.step_limit = step_limit
        self.reward_mode = conf.reward_mode

        self.reset()

    def reset(self):
        self.game.reset()
        self.done = False
        self.total_reward = 0
        self.steps = 0
        self.valid_steps = 0
        self.total_score = 0

        return process_state(self.game.board)

    def average_reward(self):
        return self.total_reward/self.steps

    def execute(self, action):

        if self.done:
            raise ValueError('This episode is finished')

        board = self.game.board

        action_name = self.action_map[action]

        if action_name == 'left':
            score = self.game.move_left()

        elif action_name == 'right':
            score = self.game.move_right()

        elif action_name == 'up':
            score = self.game.move_up()

        elif action_name == 'down':
            score = self.game.move_down()

        new_board = self.game.board

        if np.any(new_board != board):
            move_valid = True
            self.valid_steps += 1
        else:
            move_valid = False

        self.total_score += score
        if self.reward_mode == 'dense':
            if score != 0:
                score = math.log(score, 2)
            reward = score/11.0
            if not move_valid:
                reward = -1

        elif self.reward_mode == 'valid':
            if move_valid:
                reward = 1.0
            else:
                reward = -1.0

        elif self.reward_mode == 'normalized':
            if score == 0:
                reward = 0
            else:
                reward = score/np.max(new_board)

            if not move_valid:
                reward = -1

        self.done = self.game.done
        self.total_reward += reward
        self.steps += 1

        if self.done or self.steps >= self.step_limit:
            self.done = True
            return self.empty_state, reward, self.game.done
        else:
            return process_state(self.game.board), reward, self.game.done


class KeyPressHander(object):

    def __init__(self):

        conf = config.get_config()
        self.reverse_action_map = dict([(v, k) for k, v in
                                        conf.action_map.items()])
        self.env = Env2048()
        self.total_reward = 0

        os.system('clear')
        self.env.game.print_()

    def on_press(self, key):

        if key in [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up,
                   keyboard.Key.down]:
            os.system('clear')

            if key == keyboard.Key.left:
                _, r, _ = self.env.execute(self.reverse_action_map['left'])

            if key == keyboard.Key.right:
                _, r, _ = self.env.execute(self.reverse_action_map['right'])

            if key == keyboard.Key.up:
                _, r, _ = self.env.execute(self.reverse_action_map['up'])

            if key == keyboard.Key.down:
                _, r, _ = self.env.execute(self.reverse_action_map['down'])

            self.env.game.print_()
            self.total_reward += r

            print(f"Total Reward = {self.total_reward} "
                  f"Valid steps = {self.env.valid_steps}")
            if self.env.done:
                print('Episode Complete')
                exit()


if __name__ == '__main__':
    from pynput import keyboard

    kph = KeyPressHander()
    with keyboard.Listener(on_press=kph.on_press) as listener:
        listener.join()
