import os

import torch
from torch.autograd import Variable


def variable(shape, cuda=False, type_='float'):
    if type_ == 'float':
        x = torch.FloatTensor(*shape)
    elif type_ == 'long':
        x = torch.LongTensor(*shape)
    x = Variable(x)
    if cuda:
        x = x.cuda()

    return x


def make_dir(path):
    try:
        os.makedirs(path)
    except OSError:
        pass
