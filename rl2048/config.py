import argparse
import random

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--pretraining_steps', type=int, default=10**5)
parser.add_argument('--exp_buffer_size', type=int, default=10**6)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--hidden_units', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_episodes', type=int, default=10**6)
parser.add_argument('--start_random', type=float, default=1.0)
parser.add_argument('--end_random', type=float, default=0.1)
parser.add_argument('--random_anneal_steps', type=int, default=10**6)
parser.add_argument('--reward_mode', type=str, default='normalized')
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_every', type=int, default=10**4)
parser.add_argument('--validation_episodes', type=int, default=10)
parser.add_argument('--train_every', type=int, default=4)
parser.add_argument('--episode_step_limit', type=int, default=100)
parser.add_argument('--debug', type=bool, default=False)
parser.add_argument('--name', type=str, default='default')
parser.add_argument('--dump_every', type=int, default=100000)


def get_config():
    conf = parser.parse_args()

    conf.action_map = {
        0: 'left',
        1: 'right',
        2: 'up',
        3: 'down',
    }

    random.seed(conf.seed)
    np.random.seed(conf.seed)

    return conf
