import numpy as np

from torch import nn


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
