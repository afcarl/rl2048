import sys

import numpy as np
import pytest
import torch
from rl2048.env import Env2048, process_state
from rl2048.game import board_print

sys.argv = []

no_move_board = np.array(
    [[2, 4, 2, 4],
     [4, 2, 4, 2],
     [2, 4, 2, 4],
     [4, 2, 4, 2]]
)


def test_bad_state():

    state = np.arange(16)/16.0

    with pytest.raises(ValueError):
        process_state(state)


def test_process_state():

    a = np.array(
        [[ 2,   -1,  16,   4],  # noqa
         [-1,   16,   8,  32],
         [64,   64,  16,  64],
         [128, 256,  -1, 128]]
    )

    b = np.array(
        [[1./11,    -1, 4./11, 2./11],
         [   -1, 4./11, 3./11, 5./11],  # noqa
         [6./11, 6./11, 4./11, 6./11],
         [7./11, 8./11,    -1, 7./11]]
    )

    np.testing.assert_almost_equal(process_state(a), b)


@pytest.mark.parametrize('reward_mode', ['dense', 'valid'])
def test_no_move(reward_mode):

    env = Env2048()
    env.reward_mode = reward_mode

    for i in range(4):
        env.reset()
        env.game.board = no_move_board
        state, r, _ = env.execute(i)

        assert env.game.done
        np.testing.assert_almost_equal(state, 0)

        for i in range(4):
            with pytest.raises(ValueError):
                env.execute(i)


@pytest.mark.parametrize('reward_mode', ['dense', 'valid'])
def test_limit(reward_mode):

    env = Env2048()
    env.reward_mode = reward_mode
    env.step_limit = 10
    rng = np.random.RandomState(666)

    env.game.board = np.zeros((4, 4), dtype=np.int) - 1
    env.game.board[0, 0] = 2

    reward = 0
    for i in range(10):
        assert not env.done

        a = rng.randint(4)
        state, r, _ = env.execute(a)
        reward += r

    assert env.done
    assert pytest.approx(env.average_reward()) == reward/10


@pytest.mark.parametrize('reward_mode', ['dense', 'valid'])
def test_all_collapse(reward_mode):

    env = Env2048()
    env.reward_mode = reward_mode
    env.game.board = np.zeros((4, 4), dtype=np.int) + 2

    env.execute(0)
    env.game.board[:, 2:] = -1

    env.execute(2)
    env.game.board[2:, :] = env.game.board[:, 2:] = -1

    env.execute(0)
    env.game.board[2:, 1:] = env.game.board[1:, 2:] = -1

    state, _, _ = env.execute(2)

    assert pytest.approx(state[0][0]) == 5.0/11

    if reward_mode == 'valid':
        assert pytest.approx(1) == env.average_reward()
    elif reward_mode == 'dense':
        assert pytest.approx((5)/11.) == env.average_reward()


def test_reward_valid():

    env = Env2048()
    env.reward_mode = 'valid'

    rng = np.random.RandomState(42)

    valid_moves = 0
    for i in range(100):
        a = rng.randint(0, 4)

        if a == 0:
            move = env.game.can_move_left
        elif a == 1:
            move = env.game.can_move_right
        elif a == 2:
            move = env.game.can_move_up
        elif a == 3:
            move = env.game.can_move_down

        if move():
            valid_moves += 1
        else:
            valid_moves -= 1

        env.execute(a)
        if env.done:
            break

    assert pytest.approx(valid_moves/(i + 1)) == env.average_reward()


@pytest.mark.parametrize('reward_mode', ['dense', 'valid'])
def test_reward_sum(reward_mode):

    env = Env2048()
    env.reward_mode = reward_mode

    rng = np.random.RandomState(42)

    for i in range(100):
        a = rng.randint(0, 4)

        before = env.game.board[env.game.board > 0].sum()
        env.execute(a)
        after = env.game.board[env.game.board > 0].sum()

        assert after - before in [0, 2, 4]
        if env.done:
            break
