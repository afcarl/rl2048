import numpy as np

from torch import nn


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


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

    def __init__(self, input_shape, output_size, num_hidden,
                 residual_blocks=4):

        super(ConvNet, self).__init__()

        layers = [ResidualBlock(1, num_hidden)]

        for i in range(residual_blocks):
            layers.append(ResidualBlock(num_hidden, num_hidden))

        layers.append(conv3x3(num_hidden, 4))

        self.net = nn.Sequential(*layers)
        self.pool = nn.AvgPool2d(4)

    def forward(self, state):

        state = state.unsqueeze(1)
        out = self.net(state)
        pooled = self.pool(out)

        out = pooled.squeeze(3).squeeze(2)

        return out
