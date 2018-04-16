import copy
import logging
import os
import pickle
import pprint
import random
import time
from collections import namedtuple

import numpy as np
import torch
from torch import nn

from . import config, utils
from .env import Env2048
from .game import board_print
from .stats import Stats

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


def dump_file(filename, obj):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def copy_data(var, array):
    if isinstance(array, np.ndarray):
        if isinstance(var.data, torch.FloatTensor):
            array = array.astype(np.float)
        array = torch.Tensor(array)

    var.data.copy_(array)


def process_state(state):

    assert np.max(state) >= 2
    state = np.array(state, dtype=np.float, copy=True)
    positive_mask = state > 0
    positive_states = state[positive_mask]

    state[positive_mask] = np.log(positive_states)/11.0
    state[np.logical_not(positive_mask)] = -1.0

    return state


class MLP(nn.Module):

    def __init__(self, input_size, output_size, num_hidden):

        super(MLP, self).__init__()

        self.modules = []
        self.num_hidden = num_hidden

        self.net = nn.Sequential(
            nn.Linear(input_size, self.num_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_hidden, self.num_hidden),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.num_hidden, output_size)
        )

    def forward(self, state):

        return self.net(state)


def array_in_range(a, low, high):

    return np.all(np.logical_and(a >= low, a <= high))


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

        states = process_state(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        finished = np.array(finished)
        next_states = process_state(next_states)

        assert array_in_range(states, -1, 1)
        assert array_in_range(actions, 0, 3)
        assert array_in_range(rewards, -1, 1)
        assert array_in_range(finished, 0, 1)
        assert array_in_range(next_states, -1, 1)

        return states, actions, rewards, finished, next_states


class DQN(object):
    def __init__(self):
        conf = config.get_config()

        self.pretraining_steps = conf.pretraining_steps
        self.num_actions = len(conf.action_map)
        self.hidden_units = conf.hidden_units
        self.input_size = 16
        self.cuda = torch.cuda.is_available()
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
        self.dump_every = conf.dump_every

        self.exp_buffer = ExperineReplayBuffer()

        # Variables
        self.state_batch = utils.variable(
            (self.batch_size, self.input_size),
            cuda=self.cuda, type_='float')

        self.next_state_batch = utils.variable(
            (self.batch_size, self.input_size),
            cuda=self.cuda, type_='float')

        self.state_batch.data.zero_()
        self.next_state_batch.data.zero_()

        self.state = utils.variable((1, self.input_size, ), cuda=self.cuda,
                                    type_='float')

        self.y_target = utils.variable((self.batch_size, ), cuda=self.cuda,
                                       type_='float')
        self.action = utils.variable((self.batch_size, ), cuda=self.cuda,
                                     type_='long')

        self.net_target = MLP(self.input_size, self.num_actions,
                              conf.hidden_units)
        if self.cuda:
            self.net_target.cuda()

        # We optimize only the main network.
        self.net_main = copy.deepcopy(self.net_target)
        self.optimizer = torch.optim.Adam(self.net_main.parameters())

        self.criterion = nn.MSELoss()

        log_dir = os.path.join('logs', conf.name)
        os.makedirs(log_dir, exist_ok=True)

        self.stats_file = os.path.join(log_dir, 'stats.pkl')

        logging.info('DQN Initialized')

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def get_network(self, kind):
        if kind == 'main':
            net = self.net_main
        elif kind == 'target':
            net = self.net_target
        else:
            raise ValueError("Unrecognized network")

        return net

    def predict_action(self, state, kind):

        assert array_in_range(state, -1, 1)

        state = state.reshape(1, -1)
        copy_data(self.state, state)

        net = self.get_network(kind)
        net.eval()

        q = net(self.state)
        _, index = torch.max(q, 1)
        return index.data[0]

    def epsilon_greedy_main_action(self, state, epsilon):

        num = random.uniform(0, 1)

        if num < epsilon:
            return self.random_action()
        else:
            return self.predict_action(state, 'main')

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

    def predict_batch_maxq(self, states, kind):
        q_values = self.get_network(kind)(states)

        values, _ = torch.max(q_values, dim=1)
        return values.data.cpu().numpy()

    def annealed_prob(self, steps):

        prob_dec = ((self.start_random - self.end_random) * steps /
                    self.random_anneal_steps)
        prob = self.start_random - prob_dec
        return max(prob, self.end_random)

    def train(self):

        env_train = Env2048(self.episode_step_limit)
        self.stats = Stats()

        state = env_train.reset()
        episode_number = 0
        last_update_time = time.time()
        env_train = Env2048(self.episode_step_limit)

        logging.info('Starting training')

        for step in range(self.max_steps):

            action = self.epsilon_greedy_main_action(process_state(state),
                                                     self.annealed_prob(step))

            next_state, reward, done = env_train.execute(action)
            self.exp_buffer.add(state, action, reward, done, next_state)
            state = next_state

            if env_train.done:
                self.print_training_stats(episode_number, env_train)
                avg_reward = env_train.average_reward()
                state = env_train.reset()
                episode_number += 1

            if step % self.train_every == 0 and step > 0:
                batch_loss = self.sample_and_train_batch()

                if step > 1000:
                    self.stats.record('train', 'Loss', batch_loss, step)
                logging.debug(f"Step {step}: loss = {batch_loss}")

            if step % self.update_every == 0 and step > 0:
                # Update target net
                self.net_target = copy.deepcopy(self.net_main)

                update_time = time.time() - last_update_time
                self.print_validation_stats(step, update_time)
                logging.info(f"Step {step}: loss = {batch_loss} "
                             f"avg reward = {avg_reward}")

                last_update_time = time.time()

            if step % self.dump_every == 0:
                dump_file(self.stats_file, self.stats)

    def print_training_stats(self, episode_number, env_train):
        episode_number += 1
        max_block = np.max(env_train.game.board)
        avg_reward = env_train.average_reward()

        logging.debug((
            f"Episode {episode_number}: Max block = {max_block}"
            f"Avg. Reward = {avg_reward}"))

        self.stats.record('train', 'Max Block-ep', max_block,
                          episode_number)
        self.stats.record('train', 'Avg. Reward',
                          env_train.average_reward(), episode_number)

    def print_validation_stats(self, step, update_time):

        result = self.validate(self.validation_episodes)

        max_block = result['max_block']
        avg_block = result['avg_block']
        valid_frac = result['valid_frac']
        avg_reward = result['avg_reward']
        logging.info(f"Validation {step} : max block = {max_block} "
                     f"avg block = {avg_block} valid frac. = {valid_frac} "
                     f"avg reward = {avg_reward}")

        self.stats.record('val', 'Max Block-step', result['max_block'], step)
        self.stats.record('val', 'Valid Fraction', result['valid_frac'], step)

    def sample_and_train_batch(self):
        self.net_main.zero_grad()

        results = self.exp_buffer.sample(self.batch_size)
        states, actions, rewards, done, next_states = results
        copy_data(self.state_batch, states)
        copy_data(self.next_state_batch, next_states)
        copy_data(self.action, actions)

        self.net_main.train()
        # Get max q value of the next state from the target net
        q_target_max_next_state = self.predict_batch_maxq(
            self.next_state_batch, 'target')

        # Compute y based on if the episode terminates
        y = np.where(done, rewards,
                     rewards + self.gamma*q_target_max_next_state)
        copy_data(self.y_target, y)

        # Get computed q value for the given states and actions from
        # the main net
        q_s = self.net_main(self.state_batch)
        q_s_a = action_select(q_s, self.action)

        loss = self.criterion(q_s_a, self.y_target)
        loss.backward()

        self.optimizer.step()

        return loss.data[0]

    def validate(self, steps):

        env = Env2048(self.episode_step_limit)
        blocks = []
        valid_steps = 0
        total_steps = 0
        total_reward = 0
        for i in range(steps):
            state = env.reset()
            while not env.done:
                action = self.predict_action(process_state(state), 'target')
                state, r, _ = env.execute(action)
                total_reward += r

            valid_steps += env.valid_steps
            total_steps += env.steps
            blocks.append(np.max(env.game.board))

        blocks = np.array(blocks)
        return {
            'max_block': np.max(blocks),
            'avg_block': np.average(blocks.astype(np.float)),
            'valid_frac': valid_steps/total_steps,
            'avg_reward': total_reward/total_steps
        }


pprint.pprint(vars(config.get_config()), indent=2)
dqn = DQN()

dqn.pretrain()
dqn.train()
