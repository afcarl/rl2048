import sys

import numpy as np
from rl2048.exp_buffer import ExperineReplayBuffer

sys.argv = []


def test_sample():

    exp_buffer = ExperineReplayBuffer()
    exp_buffer.size = 10

    for i in range(11):
        exp_buffer.add(i, -1, -1, -1, -1)

    states, _, _, _, _ = exp_buffer.sample(100)
    assert np.all(states > 0)


def test_single():

    exp_buffer = ExperineReplayBuffer()
    exp_buffer.size = 1

    exp_buffer.add(1, -1, -1, -1, -1)
    exp_buffer.add(2, -1, -1, -1, -1)

    states, _, _, _, _ = exp_buffer.sample(100)

    assert np.all(states == 2)


def test_random_stats():

    exp_buffer = ExperineReplayBuffer()
    exp_buffer.size = 5

    for i in range(10):
        exp_buffer.add(-1, -1, -1, -1, i)

    for i in range(100):
        _, _, _, _, finished = exp_buffer.sample(100)
        values = set(np.unique(finished))
        assert values <= set(range(5, 10))


def test_less_samples():

    exp_buffer = ExperineReplayBuffer()
    exp_buffer.size = 100

    for i in range(10):
        exp_buffer.add(-1, -1, -1, -1, i)

    for i in range(100):
        _, _, _, _, finished = exp_buffer.sample(100)
        values = set(np.unique(finished))
        assert values <= set(range(10))
