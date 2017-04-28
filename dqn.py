import config
from collections import namedtuple
from env import Env2048
from game import board_print
import random
from torch import nn
import utils
import numpy as np
import logging


Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'next_state'])

logger = logging.getLogger("DQN")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s %(name)s %(asctime)s:%(message)s",
                              "%H:%M:%S")
ch.setFormatter(formatter)

logger.addHandler(ch)


# add ch to logger


class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_hidden):

        super(MLP, self).__init__()

        self.modules = []
        self.num_hidden = num_hidden

        self.linear1 = nn.Linear(input_size, self.num_hidden)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.linear2 = nn.Linear(num_hidden, self.num_hidden)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.output = nn.Linear(self.num_hidden, output_size)

    def forwrd(self, state):
        out1 = self.relu1(self.linear1(state))
        out2 = self.relu2(self.linear2(out1))

        return self.output(out2)


class ExperineReplayBuffer(object):

    def __init__(self):
        conf = config.get_config()

        self.size = conf.exp_buffer_size
        self.buffer_ = []
        self.action_map = conf.action_map

    def add(self, state, action, reward, done, next_state):

        step = Step(state=state, action=action, reward=reward, done=done,
                    next_state=next_state)
        self.buffer_.append(step)

        if len(self.buffer_) > self.size:
            self.buffer_.pop(0)

    def print_(self):

        for step in self.buffer_:
            state = step.state.reshape(4, 4)
            action = self.action_map[step.action]
            reward = step.reward
            next_state = step.next_state.reshape(4, 4)

            print('-'*20)
            board_print(state)
            print(action, reward)
            board_print(next_state)
            print()

    def sample(self, batch_size):

        indices = np.random.randint(0, len(self.buffer_), size=batch_size)

        states = []
        rewards = []
        actions = []
        next_states = []
        for idx in indices:
            step = self.buffer_[idx]
            states.append(step.state)
            rewards.append(step.reward)
            actions.append(step.action)
            next_states.append(step.next_state)

        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states))


class DQN(object):
    def __init__(self):
        conf = config.get_config()

        self.pretraining_steps = conf.pretraining_steps
        self.env = Env2048()
        self.num_actions = len(conf.action_map)
        self.hidden_units = conf.hidden_units
        self.input_size = 16
        self.cuda = conf.cuda
        self.batch_size = conf.batch_size

        self.exp_buffer = ExperineReplayBuffer()
        self.state_main = utils.float_variable((self.batch_size,
                                                self.input_size),
                                               cuda=self.cuda)

        self.state_target = utils.float_variable((self.batch_size,
                                                  self.input_size),
                                                 cuda=self.cuda)

        self.net_target = MLP(self.input_size, self.num_actions,
                              conf.hidden_units)
        self.net_main = MLP(self.input_size, self.num_actions,
                            conf.hidden_units)

        logger.info('DQN Initialized')

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def pretrain(self):

        logger.info('Pretraining started')
        state = self.env.reset()

        for i in range(self.pretraining_steps):
            action = self.random_action()
            new_state, reward, done = self.env.execute(action)

            self.exp_buffer.add(state, action, reward, done, new_state)
            state = new_state

            if self.env.done:
                state = self.env.reset()
        logger.info('Pretraining done')



dqn = DQN()

dqn.pretrain()

dqn.exp_buffer.sample(100)
