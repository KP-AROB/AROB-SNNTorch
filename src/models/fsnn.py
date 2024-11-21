import torch
import snntorch as snn
from torch import nn
from .base import BaseFSNN


class FCSNN(BaseFSNN):
    def __init__(self,
                 n_input: int = 28 * 28,
                 n_hidden: int = 16,
                 n_output: int = 10,
                 beta: float = 0.8,
                 timesteps: int = 50,
                 encoding_type: str = None):
        super().__init__(n_input, n_hidden, n_output, beta, timesteps, encoding_type)
        self.fc1 = nn.Linear(n_input**2, n_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        spk2_rec = []
        mem2_rec = []

        data = self.encoding(
            x, num_steps=self.timesteps) if self.encoding else x

        for t in range(self.timesteps):
            spike_data = data[t] if self.encoding else data
            cur1 = self.fc1(spike_data.flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
