import torch
import snntorch as snn
from snntorch import utils
from torch import nn
from .base import BaseFSNN


class S_MNET10(BaseFSNN):
    def __init__(self,
                 input_shape: tuple = (1, 224, 224),
                 n_hidden: int = 16,
                 n_output: int = 4,
                 n_steps: int = 50,
                 beta: float = 0.95,
                 encoding_type: str = None):
        super().__init__(input_shape, n_hidden, n_output,
                         n_steps, beta, encoding_type)

        k_size = 3
        n_blocks = 4
        fc_size = int(
            (input_shape[1] - n_blocks * k_size + 1 * n_blocks) / 2**n_blocks) - 1

        self.net = nn.Sequential(
            # block 1
            nn.Conv2d(input_shape[0], 16, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta),
            # block 2
            nn.Conv2d(16, 32, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta),
            # block 3
            nn.Conv2d(32, 32, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta),
            # block 4
            nn.Conv2d(32, 64, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta),
            nn.Flatten(),
            nn.Linear(64*fc_size**2, 1024),
            snn.Leaky(beta=beta),
            nn.Linear(1024, self.n_output),
            snn.Leaky(beta=beta, output=True)
        ).to(self.device)

    def forward(self, x):
        mem_rec = []
        spk_rec = []
        utils.reset(self.net)

        for _ in range(self.n_steps):
            spk_out, mem_out = self.net(x)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)
        return torch.stack(spk_rec), torch.stack(mem_rec)
