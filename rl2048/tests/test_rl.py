import numpy as np
import pytest
from rl2048.env import Env2048
from rl2048.exp_buffer import ExperineReplayBuffer


def assert_in_range(a, l, h):

    assert np.max(a) <= h
    assert np.min(a) >= l


@pytest.mark.parametrize('reward_mode', ['dense', 'valid'])
def test_random_moves(reward_mode):

    env = Env2048()
    env.step_limit = 100
    exp_buffer = ExperineReplayBuffer()
    exp_buffer.size = 10

    for i in range(100):

        state = env.reset()
        while not env.done:

            action = np.random.randint(0, 4)
            next_state, reward, done = env.execute(action)

            exp_buffer.add(state, action, reward, env.done, next_state)
            assert_in_range(state, -1, 1)
            assert_in_range(next_state, -1, 1)
            assert_in_range(action, 0, 3)
            assert_in_range(reward, -1, 1)

    state, action, reward, finished, next_state = exp_buffer.sample(100)
    assert_in_range(state, -1, 1)
    assert_in_range(action, 0, 3)
    assert_in_range(reward, -1, 1)
    assert_in_range(finished, 0, 1)
    assert_in_range(next_state, -1, 1)
