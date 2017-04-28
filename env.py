import config
from game import Game2048
import os
import numpy as np
from pynput import keyboard


class Env2048(object):

    def __init__(self):

        conf = config.get_config()
        self.game = Game2048()
        self.action_map = conf.action_map
        self.empty_state = np.zeros(16, dtype=np.int).flatten()

        self.reset()

    def reset(self):
        self.game.reset()
        self.done = False
        self.total_reward = 0

        return self.game.board.flatten()

    def execute(self, action):

        if self.done:
            raise ValueError('This episode is finished')

        board = self.game.board

        action_name = self.action_map[action]

        if action_name == 'left':
            self.game.move_left()

        elif action_name == 'right':
            self.game.move_right()

        elif action_name == 'up':
            self.game.move_up()

        elif action_name == 'down':
            self.game.move_down()

        new_baord = self.game.board

        if new_baord.max() > board.max():
            reward = 1
        else:
            reward = 0

        self.done = self.game.done
        self.total_reward += reward

        if self.done:
            return self.empty_state.flatten(), reward, self.game.done
        else:
            return new_baord.flatten(), reward, self.game.done


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

            print('Total Reward = %d' % self.total_reward)
            if self.env.done:
                print('Episode Complete')
                exit()


if __name__ == '__main__':
    kph = KeyPressHander()
    with keyboard.Listener(on_press=kph.on_press) as listener:
        listener.join()
