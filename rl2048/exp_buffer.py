import json
import logging
from collections import OrderedDict, namedtuple

import numpy as np

from . import config

Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'next_state'])

class FixedFIFO(object):

    def __init__(self, size):

        self.size = size
        self.buffer_ = [None for _ in range(self.size)]
        self.index = 0
        self.valid_samples = 0

    def add(self, item):

        self.buffer_[self.index] = item
        self.index = (self.index + 1)%self.size
        self.valid_samples = min(self.valid_samples + 1, self.size)

    def sample(self, size):
        indices = np.random.randint(0, self.valid_samples, size=size)

        return [self.buffer_[i] for i in indices]


class ExperineReplayBuffer(object):

    def __init__(self):
        conf = config.get_config()
        self.size = conf.exp_buffer_size

        self.fifo = FixedFIFO(self.size)

    def add(self, state, action, reward, done, next_state):

        step = Step(state=state, action=action, reward=reward, done=done,
                    next_state=next_state)
        self.fifo.add(step)

    def sample(self, batch_size):

        states = []
        rewards = []
        actions = []
        next_states = []
        finished = []
        for step in self.fifo.sample(batch_size):

            states.append(step.state)
            rewards.append(step.reward)
            actions.append(step.action)
            finished.append(step.done)
            next_states.append(step.next_state)

        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        finished = np.array(finished)
        next_states = np.array(next_states)

        assert np.max(states) <= 1
        assert np.max(next_states) <= 1

        return states, actions, rewards, finished, next_states

    def print_stats(self):

        n = 10
        rewards = {}
        state_max = {}
        rewards = np.array([step.reward for step
                            in self.buffer_[:self.valid_samples]])
        neg_count = np.sum(rewards < 0)
        pos_rewards = rewards[rewards >= 0]

        counts, edges = np.histogram(pos_rewards, n, (0, 1))
        reward_dict = OrderedDict([('-1', int(neg_count))])
        for c, e in zip(counts, edges):
            reward_dict[round(e, 2)] = int(c)

        reward_str = json.dumps(reward_dict, indent=2)
        return logging.info(f"size = {self.valid_samples} "
                            f"Rewards {reward_str}")
