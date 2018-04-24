import copy
import logging
import os
import pickle
import pprint
import random
import time

import numpy as np
import torch
from torch import nn

from . import config, utils
from .env import Env2048
from .exp_buffer import ExperineReplayBuffer
from .stats import Stats
from .utils import action_select, copy_data

if config.get_config().debug:
    level = logging.DEBUG
else:
    level = logging.INFO

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s : %(message)s',
    level=level,
)


def dump_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


class MLP(nn.Module):

    def __init__(self, input_shape, output_size, num_hidden):

        super(MLP, self).__init__()

        self.num_hidden = num_hidden

        self.input_size = int(np.prod(input_shape))
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.num_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_hidden, output_size)
        )

    def forward(self, state):

        state = state.view(-1, self.input_size)
        return self.net(state)


class ConvNet(nn.Module):

    def __init__(self, input_shape, output_size, num_hidden):

        super(ConvNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, num_hidden, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden, num_hidden, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden, num_hidden, 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden, 4, 1),
        )

    def forward(self, state):

        state = state.unsqueeze(1)
        out = self.net(state)
        out = out.squeeze(3).squeeze(2)

        return out


class DQN(object):
    def __init__(self):
        conf = config.get_config()

        self.pretraining_steps = conf.pretraining_steps
        self.num_actions = len(conf.action_map)
        self.hidden_units = conf.hidden_units
        self.cuda = torch.cuda.is_available()
        self.batch_size = conf.batch_size
        self.max_episodes = conf.max_episodes
        self.start_random = conf.start_random
        self.end_random = conf.end_random
        self.random_anneal_steps = conf.random_anneal_steps
        self.gamma = conf.gamma
        self.validation_episodes = conf.validation_episodes
        self.episode_step_limit = conf.episode_step_limit
        self.train_every = conf.train_every
        self.update_every = conf.update_every
        self.dump_every = conf.dump_every
        self.num_steps = 0
        self.num_episodes = 0

        self.exp_buffer = ExperineReplayBuffer()

        # Variables
        self.state_batch = utils.variable((self.batch_size, 4, 4),
                                          cuda=self.cuda, type_='float')

        self.next_state_batch = utils.variable(
            (self.batch_size, 4, 4),
            cuda=self.cuda, type_='float')

        self.state_batch.data.zero_()
        self.next_state_batch.data.zero_()

        self.state = utils.variable((1, 4, 4), cuda=self.cuda,
                                    type_='float')

        self.y_target = utils.variable((self.batch_size, ), cuda=self.cuda,
                                       type_='float')
        self.action = utils.variable((self.batch_size, ), cuda=self.cuda,
                                     type_='long')

        self.net_main = MLP((4, 4), self.num_actions, conf.hidden_units)
        if self.cuda:
            self.net_main.cuda()

        self.optimizer = torch.optim.Adam(self.net_main.parameters())

        self.criterion = nn.MSELoss()

        log_dir = os.path.join('logs', conf.name)
        os.makedirs(log_dir, exist_ok=True)

        self.stats_file = os.path.join(log_dir, 'stats.pkl')

        logging.info('DQN Initialized')

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def predict_action(self, state):

        assert np.max(state) <= 1.0
        assert np.min(state) >= -1.0

        copy_data(self.state, state)

        self.net_main.eval()

        q = self.net_main(self.state)
        _, index = torch.max(q, 1)
        return index.data[0]

    def epsilon_greedy_main_action(self, state, epsilon):

        num = random.uniform(0, 1)

        if num < epsilon:
            return self.random_action()
        else:
            return self.predict_action(state)

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

    def predict_batch_maxq(self, states):
        q_values = self.net_main(states)

        values, _ = torch.max(q_values, dim=1)
        return values.data.cpu().numpy()

    def should_train(self):

        return (self.num_steps % self.train_every) == 0 and self.num_steps > 0

    def should_update(self):

        return self.num_steps % self.update_every == 0 and self.num_steps > 0

    def annealed_prob(self, steps):

        prob_dec = ((self.start_random - self.end_random) * steps /
                    self.random_anneal_steps)
        prob = self.start_random - prob_dec
        return max(prob, self.end_random)

    def train(self):

        env_train = Env2048(self.episode_step_limit)
        self.stats = Stats()
        logging.info('Starting training')

        for self.num_episodes in range(self.max_episodes):

            self.run_episode_train(env_train)

    def run_episode_train(self, env_train):

        state = env_train.reset()

        while not env_train.done:
            prob = self.annealed_prob(self.num_steps)
            action = self.epsilon_greedy_main_action(state, prob)
            next_state, reward, done = env_train.execute(action)

            self.num_steps += 1
            self.exp_buffer.add(state, action, reward, done, next_state)
            state = next_state

            if self.should_train():
                batch_loss = self.sample_and_train_batch()

            if self.should_update():
                self.validate()

            if self.num_steps % self.dump_every == 0:
                dump_file(self.stats_file, self.stats)

        avg_reward = env_train.average_reward()
        valid_frac = float(env_train.valid_steps)/env_train.steps
        self.stats.record('train', 'Loss',
                          batch_loss, self.num_steps)
        self.stats.record('train', 'Avg-Reward',
                          avg_reward, self.num_steps)
        self.stats.record('train', 'Valid-Fraction',
                          valid_frac, self.num_steps)

    def sample_and_train_batch(self):
        self.net_main.zero_grad()

        results = self.exp_buffer.sample(self.batch_size)
        states, actions, rewards, done, next_states = results
        copy_data(self.state_batch, states)
        copy_data(self.next_state_batch, next_states)
        copy_data(self.action, actions)

        self.net_main.train()

        copy_data(self.y_target, rewards)

        q_s = self.net_main(self.state_batch)
        q_s_a = action_select(q_s, self.action)

        loss = self.criterion(q_s_a, self.y_target)
        loss.backward()

        self.optimizer.step()

        return loss.data[0]

    def validate(self):

        env = Env2048(self.episode_step_limit)
        blocks = []
        valid_steps = 0
        total_steps = 0
        total_reward = 0
        for i in range(self.validation_episodes):
            state = env.reset()
            while not env.done:
                action = self.predict_action(state)
                state, r, _ = env.execute(action)
                total_reward += r

            valid_steps += env.valid_steps
            total_steps += env.steps
            blocks.append(np.max(env.game.board))

        blocks = np.array(blocks)

        valid_frac = valid_steps/total_steps
        avg_reward = total_reward/total_steps

        self.stats.record('valid', 'Avg-Reward',
                          avg_reward, self.num_steps)
        self.stats.record('valid', 'Valid-Fraction',
                          valid_frac, self.num_steps)
        logging.info(f"Valid {self.num_steps}: "
                     f"avg reward = {avg_reward:.2f} "
                     f"valid frac. = {valid_frac:.2f}")


pprint.pprint(vars(config.get_config()), indent=2)
dqn = DQN()

dqn.pretrain()
dqn.train()
