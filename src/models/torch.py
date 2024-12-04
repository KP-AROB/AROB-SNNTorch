from torch import nn
import torch.nn.functional as F
from src.utils.net import make_vgg_conv_block


class VGG11(nn.Module):
    def __init__(self, n_input=3, n_output=4, in_size=224):
        super(VGG11, self).__init__()
        config = [64, 'M', 128, 'M', 256, 256,
                  'M', 512, 512, 'M', 512, 512, 'M']

        self.n_input = n_input
        self.n_output = n_output
        self.in_size = in_size

        self.net = make_vgg_conv_block(config, self.n_input)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512, 1028),
            nn.ReLU(),
            nn.Linear(1028, self.n_output)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x


class VGG16(nn.Module):
    def __init__(self, n_input=3, n_output=4, in_size=224):
        super(VGG16, self).__init__()
        config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                  'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

        self.n_input = n_input
        self.n_output = n_output
        self.in_size = in_size

        self.net = make_vgg_conv_block(config, self.n_input)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7*7*512, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.n_output)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x


class MNET10(nn.Module):
    def __init__(self, n_input=3, n_output=4, in_size=224):
        super(MNET10, self).__init__()

        self.n_input = n_input
        self.n_output = n_output
        self.in_size = in_size
        k_size = 3
        n_conv = 3
        fc_size = int(
            (in_size - n_conv * k_size + 1 * n_conv) / 2**n_conv) - 1

        self.conv1 = nn.Conv2d(n_input, 16, k_size)
        self.norm1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, k_size)
        self.norm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, k_size)
        self.norm3 = nn.BatchNorm2d(64)

        self.activation = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_size * fc_size * 64, 512)
        self.fc2 = nn.Linear(512, n_output)
        self.dropout = nn.Dropout(0.4)
        self.ic_drop = nn.Dropout(0.01)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.norm1(self.activation(self.conv1(x))))
        x = self.pool(self.norm2(self.activation(self.conv2(x))))
        x = self.pool(self.norm3(self.activation(self.conv3(x))))
        x = self.flatten(x)
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
