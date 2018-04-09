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

from . import config
from . import utils
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

        self.state = utils.variable((self.input_size, ), cuda=self.cuda,
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
        self.optimizer = torch.optim.Adam(self.net_main.parameters(),
                                          lr=0.0001)

        self.criterion = nn.MSELoss()

        log_dir = os.path.join('logs', conf.name)
        os.makedirs(log_dir, exist_ok=True)

        self.stats_file = os.path.join(log_dir, 'stats.pkl')

        logging.info('DQN Initialized')

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def predict_target_single(self, state):

        copy_data(self.state, state)
        q = self.net_target(self.state)
        _, index = torch.max(q, 0)
        return index.data[0]

    def predict_main_single(self, state):

        copy_data(self.state, state)
        q = self.net_main(self.state)
        _, index = torch.max(q, 0)
        return index.data[0]

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

            action = self.epsilon_greedy_main_action(state,
                                                     self.annealed_prob(step))

            next_state, reward, done = env_train.execute(action)
            self.exp_buffer.add(state, action, reward, done, next_state)

            if env_train.done:
                self.print_training_stats(episode_number, env_train)
                env_train.reset()
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
                logging.info(f"Step {step}: loss = {batch_loss}")

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
        logging.info(
            f"Step {step}: Updating target {update_time} secs "
            "since last update")

        result = self.validate(self.validation_episodes)

        max_block = result['max_block']
        avg_block = result['avg_block']
        valid_frac = result['valid_frac']
        logging.info(f"Validation {step} : max block = {max_block} "
                     f"avg block = {avg_block} valid frac. = {valid_frac}")

        self.stats.record('val', 'Max Block-step', result['max_block'], step)
        self.stats.record('val', 'Valid Fraction', result['valid_frac'], step)

    def sample_and_train_batch(self):
        self.net_main.zero_grad()

        results = self.exp_buffer.sample(self.batch_size)
        states, actions, rewards, done, next_states = results
        copy_data(self.state_batch, states)
        copy_data(self.next_state_batch, next_states)
        copy_data(self.action, actions)

        # Get max q value of the next state from the target net
        q_target_max_next_state = self.predict_target_batch(
            self.next_state_batch)

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
            'valid_frac': valid_steps/total_steps,
        }


pprint.pprint(vars(config.get_config()), indent=2)
dqn = DQN()

dqn.pretrain()
dqn.train()
