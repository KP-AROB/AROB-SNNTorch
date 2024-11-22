import torch
from torch import nn
import torch.nn.functional as F


class FMNIST_CNN(nn.Module):
    def __init__(self, n_input=3, n_output=2):
        super(FMNIST_CNN, self).__init__()

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(n_input, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ShallowCNN(nn.Module):
    def __init__(self, n_input=3, n_output=2):
        super(ShallowCNN, self).__init__()

        self.n_fc = 29
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.conv1 = nn.Conv2d(n_input, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * self.n_fc * self.n_fc, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_output)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * self.n_fc * self.n_fc)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
