import torch
import snntorch as snn
from abc import ABC
from torch import nn
from snntorch import surrogate


class BaseCSNN(ABC):
    def __init__(self,
                 in_shape: tuple = (1, 28, 28),
                 n_class: int = 10,
                 start_features: int = 16,
                 beta: float = 0.5,
                 num_steps: int = 50,
                 k_size: int = 5):
        self.in_shape = in_shape
        self.n_class = n_class
        self.start_features = start_features
        self.beta = beta
        self.k_size = k_size
        self.spike_grad = surrogate.fast_sigmoid()
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = None


class CSNN_2C(BaseCSNN):
    def __init__(self,
                 in_shape: tuple = (1, 28, 28),
                 n_class: int = 10,
                 start_features: int = 16,
                 beta: float = 0.5,
                 k_size: int = 5):
        super().__init__(in_shape, n_class, start_features, beta, k_size)

        fc_features = int((in_shape[1] - 2 * k_size + 1 * 2) / 4) - 1

        self.net = nn.Sequential(
            nn.Conv2d(in_shape[0], start_features, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(start_features, start_features * 2, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(start_features * 2 * fc_features**2, n_class),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, output=True)).to(self.device)


class CSNN_3C(BaseCSNN):
    def __init__(self,
                 in_shape: tuple = (1, 28, 28),
                 n_class: int = 10,
                 start_features: int = 16,
                 beta: float = 0.5,
                 k_size: int = 5):
        super().__init__(in_shape, n_class, start_features, beta, k_size)

        fc_features = int((in_shape[1] - 2 * k_size + 1 * 2) / 4) - 1

        self.net = nn.Sequential(
            nn.Conv2d(in_shape[0], start_features, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(start_features, start_features*2, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(start_features*2, start_features*4, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(start_features*4*fc_features**2, n_class),
            snn.Leaky(beta=beta, spike_grad=self.spike_grad, init_hidden=True, output=True)).to(self.device)
