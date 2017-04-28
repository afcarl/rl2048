from __future__ import print_function
import numpy as np
import random
import os


start_digits = [2, 4]


def compress(values):

    assert isinstance(values, list)

    l = len(values)
    if l <= 1:
        return values, 0

    else:
        if values[0] == values[1]:
            rest, score = compress(values[2:])
            return [values[0]*2] + rest, values[0]*2 + score
        else:
            if len(values) >= 3 and values[1] == values[2]:
                rest, score = compress(values[3:])
                return [values[0], values[1]*2] + rest, score + values[1]*2
            else:
                rest, score = compress(values[2:])
                return values[:2] + rest, score


def board_left(board):

    board_return = np.zeros_like(board)
    total_score = 0
    for row in range(4):
        values = board[row]
        values = list(values[values > 0])
        values, score = compress(values)
        total_score += score

        l = len(values)
        board_return[row, :l] = values
        board_return[row, l:] = -1

    return board_return, total_score


def board_right(board):
    board = np.fliplr(board)
    board, score = board_left(board)
    return np.fliplr(board), score


def board_up(board):
    board, score = board_left(np.rot90(board, 1))
    return np.rot90(board, -1), score


def board_down(board):
    board, score = board_left(np.rot90(board, -1))
    return np.rot90(board, 1), score


def board_print(board):
    for r in range(4):
        for c in range(4):
            if board[r, c] > 0:
                print(('%d ' % board[r, c]).center(5), end='')
            else:
                print('____ ', end='')
        print()


class Game2048(object):

    def __init__(self):
        self.board = np.zeros((4, 4), dtype=np.int)
        self.reset()

    def reset(self):
        self.board[:] = -1
        self.put_random_new_number()
        self.put_random_new_number()
        self.done = False
        self.total_score = 0

    def put_random_new_number(self):
        idx = self.random_free_index()

        if len(idx) > 0:
            self.board[idx] = random.choice(start_digits)

    def random_free_index(self):
        return tuple(random.choice(np.argwhere(self.board < 0)))

    def post_move(self, board, score):
        if np.any(board != self.board):
            self.board = board
            self.put_random_new_number()
            self.total_score += score

        if not self.can_move():
            self.done = True

    def can_move_left(self):
        return np.any(board_left(self.board)[0] != self.board)

    def can_move_right(self):
        return np.any(board_right(self.board)[0] != self.board)

    def can_move_up(self):
        return np.any(board_up(self.board)[0] != self.board)

    def can_move_down(self):
        return np.any(board_down(self.board)[0] != self.board)

    def can_move(self):
        if np.any(self.board == -1):
            return True
        else:
            return (self.can_move_left() or self.can_move_right() or
                    self.can_move_up() or self.can_move_down())

    def move_left(self):
        board, score = board_left(self.board)
        return self.post_move(board, score)

    def move_right(self):
        board, score = board_right(self.board)
        return self.post_move(board, score)

    def move_up(self):
        board, score = board_up(self.board)
        return self.post_move(board, score)

    def move_down(self):
        board, score = board_down(self.board)
        return self.post_move(board, score)

    def print_(self):
        board_print(self.board)


class KeyPressHander(object):

    def __init__(self):
        self.game = Game2048()

        os.system('clear')
        self.game.print_()

    def on_press(self, key):

        if key in [keyboard.Key.left, keyboard.Key.right, keyboard.Key.up,
                   keyboard.Key.down]:
            os.system('clear')

            if key == keyboard.Key.left:
                self.game.move_left()

            if key == keyboard.Key.right:
                self.game.move_right()

            if key == keyboard.Key.up:
                self.game.move_up()

            if key == keyboard.Key.down:
                self.game.move_down()

            self.game.print_()

            print('Score = %d' % self.game.total_score)
            if self.game.done:
                print('Game Over!')
                exit()


if __name__ == '__main__':
    from pynput import keyboard

    kph = KeyPressHander()
    with keyboard.Listener(on_press=kph.on_press) as listener:
        listener.join()
