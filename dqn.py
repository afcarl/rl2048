import copy
import logging
import pprint
import random
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

import config
import utils
from env import Env2048
from game import board_print

Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'next_state'])

if config.get_config().debug:
    level = logging.DEBUG
else:
    level = logging.INFO

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s : %(message)s',
    level=level,
)


def action_select(q, a):
    return q.gather(1, a.view(-1, 1))[:, 0]


def copy_data(var, array):
    if isinstance(array, np.ndarray):
        if isinstance(var.data, torch.FloatTensor):
            array = array.astype(np.float)
        array = torch.Tensor(array)

    var.data.copy_(array)


class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_hidden):

        super(MLP, self).__init__()

        self.modules = []
        self.num_hidden = num_hidden

        self.linear1 = nn.Linear(input_size, self.num_hidden)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

        self.linear2 = nn.Linear(num_hidden, self.num_hidden)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

        self.linear3 = nn.Linear(num_hidden, self.num_hidden)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)

        self.linear4 = nn.Linear(num_hidden, self.num_hidden)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)

        self.output = nn.Linear(self.num_hidden, output_size)

    def forward(self, state):
        out1 = self.relu1(self.linear1(state))
        out2 = self.relu2(self.linear2(out1))
        out3 = self.relu3(self.linear3(out2))
        out4 = self.relu4(self.linear4(out3))

        return self.output(out4)


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
        finished = []
        for idx in indices:
            step = self.buffer_[idx]
            states.append(step.state)
            rewards.append(step.reward)
            actions.append(step.action)
            finished.append(step.done)
            next_states.append(step.next_state)

        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(finished), np.array(next_states))


class DQN(object):
    def __init__(self):
        conf = config.get_config()

        self.pretraining_steps = conf.pretraining_steps
        self.num_actions = len(conf.action_map)
        self.hidden_units = conf.hidden_units
        self.input_size = 16
        self.cuda = conf.cuda
        self.batch_size = conf.batch_size
        self.max_steps = conf.max_steps
        self.start_random = conf.start_random
        self.end_random = conf.end_random
        self.random_anneal_steps = conf.random_anneal_steps
        self.gamma = conf.gamma
        self.update_every = conf.update_every
        self.validation_episodes = conf.validation_episodes
        self.episode_step_limit = conf.episode_step_limit
        self.train_every = conf.train_every

        self.exp_buffer = ExperineReplayBuffer()
        self.state_var = utils.variable((self.batch_size, self.input_size),
                                        cuda=self.cuda, type_='float')
        self.next_state_var = utils.variable(
            (self.batch_size, self.input_size),
            cuda=self.cuda, type_='float')

        self.state_var.data.zero_()
        self.next_state_var.data.zero_()

        self.y_target = utils.variable((self.batch_size, ), cuda=self.cuda,
                                       type_='float')
        self.action_var = utils.variable((self.batch_size, ), cuda=self.cuda,
                                         type_='long')

        self.net_target = MLP(self.input_size, self.num_actions,
                              conf.hidden_units)
        if self.cuda:
            self.net_target.cuda()

        # We optimize only the main network.
        self.net_main = copy.deepcopy(self.net_target)
        self.optimizer = torch.optim.Adam(self.net_main.parameters())

        self.criterion = nn.MSELoss()

        logging.info('DQN Initialized')

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def predict_target_single(self, state):

        copy_data(self.state_var[0, :], state)
        q_first = self.net_target(self.state_var)[0]
        return np.argmax(q_first)

    def predict_main_single(self, state):

        copy_data(self.state_var[0, :], state)
        q_first = self.net_main(self.state_var)[0]
        return np.argmax(q_first)

    def epsilon_greedy_main_action(self, state, epsilon):

        num = random.uniform(0, 1)

        if num < epsilon:
            return self.random_action()
        else:
            return self.predict_main_single(state)

    def pretrain(self):

        logging.info('Pretraining started')
        env_pretrain = Env2048(self.episode_step_limit)
        state = env_pretrain.reset()

        for i in range(self.pretraining_steps):
            action = self.random_action()
            new_state, reward, done = env_pretrain.execute(action)

            self.exp_buffer.add(state, action, reward, done, new_state)
            state = new_state

            if env_pretrain.done:
                state = env_pretrain.reset()
        logging.info('Pretraining done')

    def predict_target_batch(self, states):
        q_values = self.net_target(states)

        values, _ = torch.max(q_values, dim=1)
        values = values[:, 0]
        return values.data.cpu().numpy()

    def train(self):

        env_train = Env2048(self.episode_step_limit)

        state = env_train.reset()
        random_prob = self.start_random
        episode_number = 0
        last_update_time = time.time()
        env_train = Env2048(self.episode_step_limit)

        logging.info('Starting training')
        for step in range(self.max_steps):

            random_prob -= ((self.start_random - self.end_random) /
                            self.random_anneal_steps)
            random_prob = max(random_prob, self.end_random)
            action = self.epsilon_greedy_main_action(state, random_prob)

            next_state, reward, done = env_train.execute(action)
            self.exp_buffer.add(state, action, reward, done, next_state)

            if step % self.train_every == 0:
                batch_loss = self.sample_and_train_batch()

            if env_train.done:
                episode_number += 1
                max_block = np.max(env_train.game.board)
                logging.debug('Episode %d: Max block = %d Total Reward = %d '
                              'loss = %f', episode_number, max_block,
                              env_train.total_reward, batch_loss)
                env_train.reset()

            if step % self.update_every == 0 and step > 0:
                self.net_target = copy.deepcopy(self.net_main)
                logging.info('Step %d: Updating target %f secs since last '
                             'update',
                             step, (time.time() - last_update_time))
                last_update_time = time.time()
                result = self.validate(self.validation_episodes)
                logging.info('Validation {step} : max block = {max_block} '
                             'avg block = {avg_block} '
                             'valid steps {valid_steps}/{total_steps}'
                             .format(step=step, **result))

    def sample_and_train_batch(self):
        self.net_main.zero_grad()

        results = self.exp_buffer.sample(self.batch_size)
        states, actions, rewards, done, next_states = results
        copy_data(self.state_var, states)
        copy_data(self.next_state_var, next_states)
        copy_data(self.action_var, actions)

        # Get max q value of the next state from the target net
        q_target_max_next_state = self.predict_target_batch(
            self.next_state_var)

        # Compute y based on if the episode terminates
        y = np.where(done, rewards,
                     rewards + self.gamma*q_target_max_next_state)
        copy_data(self.y_target, y)

        # Get computed q value for the given states and actions from
        # the main net
        q_s = self.net_main(self.state_var)
        q_s_a = action_select(q_s, self.action_var)

        loss = self.criterion(q_s_a, self.y_target)
        loss.backward()

        self.optimizer.step()

        return loss.data[0]

    def validate(self, steps):

        env = Env2048(self.episode_step_limit)
        blocks = []
        valid_steps = 0
        total_steps = 0
        for i in range(steps):
            state = env.reset()
            while not env.done:
                action = self.predict_target_single(state)
                state, _, _ = env.execute(action)

            valid_steps += env.valid_steps
            total_steps += env.steps
            blocks.append(np.max(env.game.board))

        blocks = np.array(blocks)
        return {
            'max_block': np.max(blocks),
            'avg_block': np.average(blocks.astype(np.float)),
            'total_steps': total_steps,
            'valid_steps': valid_steps
        }


pprint.pprint(vars(config.get_config()), indent=2)
dqn = DQN()

dqn.pretrain()
dqn.train()
