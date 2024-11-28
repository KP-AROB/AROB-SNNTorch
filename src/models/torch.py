from torch import nn
import torch.nn.functional as F


class MNET10(nn.Module):
    def __init__(self, n_input=3, n_output=4, in_size=224):
        super(MNET10, self).__init__()

        self.input = n_input
        self.n_output = n_output
        self.in_size = in_size
        k_size = 3
        n_conv = 4
        fc_size = int(
            (in_size - n_conv * k_size + 1 * n_conv) / 2**n_conv) - 1
        self.conv1 = nn.Conv2d(n_input, 16, k_size)
        self.conv2 = nn.Conv2d(16, 32, k_size)
        self.conv3 = nn.Conv2d(32, 32, k_size)
        self.conv4 = nn.Conv2d(32, 64, k_size)
        self.prelu = nn.PReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(fc_size * fc_size * 64, 1024)
        self.fc2 = nn.Linear(1024, n_output)
        self.dropout = nn.Dropout(0.5)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(self.prelu(self.conv1(x)))
        x = self.pool(self.prelu(self.conv2(x)))
        x = self.pool(self.prelu(self.conv3(x)))
        x = self.pool(self.prelu(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=1)
