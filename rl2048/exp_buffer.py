from collections import namedtuple

import numpy as np

from . import config

Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'next_state'])


class ExperineReplayBuffer(object):

    def __init__(self):
        conf = config.get_config()

        self.size = conf.exp_buffer_size
        self.buffer_ = [None for _ in range(self.size)]
        self.index = 0
        self.valid_samples = 0

    def add(self, state, action, reward, done, next_state):

        step = Step(state=state, action=action, reward=reward, done=done,
                    next_state=next_state)
        self.buffer_[self.index] = step

        self.index = (self.index + 1) % self.size
        self.valid_samples = min(self.valid_samples + 1, self.size)

    def sample(self, batch_size):

        indices = np.random.randint(0, self.valid_samples, size=batch_size)

        states = []
        rewards = []
        actions = []
        next_states = []
        finished = []
        for idx in indices:
            step = self.buffer_[idx]

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

        return states, actions, rewards, finished, next_states
