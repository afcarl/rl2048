import json
import logging
import random
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


def round_reward(r):

    return int(r*10)/10.0

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
        buffer_ = self.fifo.buffer_
        samples = self.fifo.valid_samples
        rewards = np.array([step.reward for step in buffer_[:samples]])
        neg_count = np.sum(rewards < 0)
        pos_rewards = rewards[rewards >= 0]

        counts, edges = np.histogram(pos_rewards, n, (0, 1))
        reward_dict = OrderedDict([('-1', int(neg_count))])
        for c, e in zip(counts, edges):
            reward_dict[round(e, 2)] = int(c)

        reward_str = json.dumps(reward_dict, indent=2)
        return logging.info(f"size = {self.fifo.valid_samples} "
                            f"Rewards {reward_str}")


class ResampledBuffer(object):

    def __init__(self):
        conf = config.get_config()
        self.size = conf.exp_buffer_size
        self.fifo_size = int(self.size/11.0)
        self.fifo_map = {-1: FixedFIFO(self.fifo_size)}

        for i in range(10):

            k = round(i/10.0, 2)
            self.fifo_map[k] = FixedFIFO(self.fifo_size)

    def add(self, state, action, reward, done, next_state):

        step = Step(state=state, action=action, reward=reward, done=done,
                    next_state=next_state)

        r = round_reward(reward)
        self.fifo_map[r].add(step)

    def sample(self, batch_size):

        states = []
        rewards = []
        actions = []
        next_states = []
        finished = []

        for i in range(batch_size):

            non_zero_keys = [k for k in self.fifo_map if 
                             self.fifo_map[k].valid_samples > 0]
            k = random.choice(non_zero_keys)
            fifo = self.fifo_map[k]
            idx = np.random.randint(0, fifo.valid_samples)
            step = fifo.buffer_[idx]
            
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

        reward_dict = OrderedDict()
        for k in sorted(self.fifo_map.keys()):
            reward_dict[k] = self.fifo_map[k].valid_samples

        reward_str = json.dumps(reward_dict, indent=2)
        return logging.info(f"Rewards {reward_str}")
