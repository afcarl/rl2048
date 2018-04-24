import numpy as np
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


def copy_data(var, array):

    if isinstance(array, np.ndarray):
        if isinstance(var.data, (torch.FloatTensor, torch.cuda.FloatTensor)):
            tensor = torch.FloatTensor(array)
        elif isinstance(var.data, (torch.LongTensor, torch.cuda.LongTensor)):
            tensor = torch.LongTensor(array)
        else:
            raise ValueError(f"Unknown variable tensor type {type(var.data)}.")

    var.data.copy_(tensor)


def action_select(q, a):
    return q.gather(1, a.view(-1, 1))[:, 0]
