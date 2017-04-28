import torch
from torch.autograd import Variable


def float_variable(shape, cuda=False):
    x = torch.FloatTensor(shape)
    x = Variable(x)
    if cuda:
        x = x.cuda()

    return x
