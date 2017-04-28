import config
from collections import namedtuple
from env import Env2048
from game import board_print
import random
from torch import nn
import torch
import utils
import numpy as np
import logging
import copy
import time


Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'next_state'])

logger = logging.getLogger("DQN")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(levelname)s %(name)s %(asctime)s:%(message)s",
                              "%H:%M:%S")
ch.setFormatter(formatter)

logger.addHandler(ch)


def action_select(q, a):
    return q.gather(1, a.view(-1, 1))[:, 0]


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

    def forward(self, state):
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
        self.validate_every = conf.validate_every
        self.validation_episodes = conf.validation_episodes
        self.episode_step_limit = conf.episode_step_limit

        self.exp_buffer = ExperineReplayBuffer()
        self.state_input = utils.variable((self.batch_size, self.input_size),
                                          cuda=self.cuda, type_='float')
        self.state_input.data.zero_()

        self.y_target = utils.variable((self.batch_size, ), cuda=self.cuda,
                                       type_='float')
        self.action_input = utils.variable((self.batch_size, ), cuda=self.cuda,
                                           type_='long')

        self.net_target = MLP(self.input_size, self.num_actions,
                              conf.hidden_units)
        if self.cuda:
            self.net_target.cuda()

        self.net_main = copy.deepcopy(self.net_target)
        self.optimizer = torch.optim.Adam(self.net_target.parameters())

        self.criterion = nn.MSELoss()

        logger.info('DQN Initialized')

    def random_action(self):
        return random.randint(0, self.num_actions - 1)

    def predict_target_single(self, state):

        self.state_input.data[0, :] = torch.Tensor(state.astype(np.float))
        q_first = self.net_target(self.state_input)[0]
        return np.argmax(q_first)

    def predict_main_single(self, state):

        self.state_input.data[0, :] = torch.Tensor(state.astype(np.float))
        q_first = self.net_main(self.state_input)[0]
        return np.argmax(q_first)

    def epsilon_greedy_main_action(self, state, epsilon):

        num = random.uniform(0, epsilon)

        if num < epsilon:
            return self.random_action()
        else:
            return self.predict_main_single(state)

    def pretrain(self):

        logger.info('Pretraining started')
        env_pretrain = Env2048(self.episode_step_limit)
        state = env_pretrain.reset()

        for i in range(self.pretraining_steps):
            action = self.random_action()
            new_state, reward, done = env_pretrain.execute(action)

            self.exp_buffer.add(state, action, reward, done, new_state)
            state = new_state

            if env_pretrain.done:
                state = env_pretrain.reset()
        logger.info('Pretraining done')

    def predict_main_batch(self, states_var):
        q_values = self.net_main(states_var)

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

        for step in range(self.max_steps):

            random_prob -= ((self.start_random - self.end_random) /
                            self.random_anneal_steps)
            random_prob = max(random_prob, self.end_random)
            action = self.epsilon_greedy_main_action(state, random_prob)

            next_state, reward, done = env_train.execute(action)
            self.exp_buffer.add(state, action, reward, done, next_state)

            batch_loss = self.sample_and_train_batch()

            if env_train.done:
                episode_number += 1
                max_block = np.max(env_train.game.board)
                logger.info('Episode %d: Max block = %d Total Reward = %d '
                            'loss = %f', episode_number, max_block,
                            env_train.total_reward, batch_loss)
                env_train.reset()

            if step % self.update_every == 0 and step > 0:
                self.net_main = copy.deepcopy(self.net_target)
                logger.info('Step %d: Updating main %f secs since last update',
                            step, (time.time() - last_update_time))
                last_update_time = time.time()

            if step % self.validate_every == 0 and step > 0:
                blocks = self.validate(self.validation_episodes)
                logger.info('Validation: max blocks = %d, avg block = %d',
                            np.max(blocks), np.average(blocks))

    def sample_and_train_batch(self):
        self.net_target.zero_grad()

        results = self.exp_buffer.sample(self.batch_size)
        states, actions, rewards, done, next_states = results
        states = torch.Tensor(states.astype(np.float))
        actions = torch.LongTensor(actions)
        self.state_input.data.copy_(states)
        self.action_input.data.copy_(actions)

        q_main_max = self.predict_main_batch(self.state_input)

        y = np.where(done, rewards, rewards + self.gamma*q_main_max)
        self.y_target.data.copy_(torch.Tensor(y))

        q_s = self.net_target(self.state_input)
        q_s_a = action_select(q_s, self.action_input)

        loss = self.criterion(q_s_a, self.y_target)
        loss.backward()

        self.optimizer.step()

        return loss.data[0]

    def validate(self, steps):

        env = Env2048(self.episode_step_limit)
        blocks = []
        for i in range(steps):
            state = env.reset()
            while not env.done:
                action = self.predict_target_single(state)
                state, _, _ = env.execute(action)

            blocks.append(np.max(env.game.board))

        return np.array(blocks)


dqn = DQN()

dqn.pretrain()
dqn.train()
