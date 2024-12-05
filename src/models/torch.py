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

        self.block1 = nn.Sequential(
            nn.Conv2d(n_input, 16, k_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.05),
            nn.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, k_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(0.05),
            nn.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, k_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05),
            nn.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, k_size),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.05),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_output)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x
