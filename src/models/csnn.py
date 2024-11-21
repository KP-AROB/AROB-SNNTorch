import torch
import snntorch as snn
from torch import nn
from .base import BaseFSNN


class CBIS_CSNN(BaseFSNN):
    def __init__(self,
                 n_input: int = 1,
                 n_hidden: int = 16,
                 n_output: int = 2,
                 beta: float = 0.95,
                 timesteps: int = 50,
                 image_size: int = 64,
                 encoding_type: str = None):
        super().__init__(n_input, n_hidden, n_output, beta, timesteps, encoding_type)

        n_conv = 2
        k_size = 5
        fc_features = int(
            (image_size - n_conv * k_size + 1 * n_conv) / 2**n_conv) - 1

        self.conv1 = nn.Conv2d(n_input, n_hidden, k_size)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(n_hidden, n_hidden*2, k_size)
        self.lif2 = snn.Leaky(beta=beta)

        self.fc1 = nn.Linear(n_hidden*2 * fc_features**2, n_hidden*2)
        self.lif3 = snn.Leaky(beta=beta)

        self.fc_out = nn.Linear(100, n_output)
        self.lif_out = snn.Leaky(beta=beta)

        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem_out = self.lif_out.reset_mem()

        spk_out_rec = []
        mem_out_rec = []

        data = self.encoding(
            x, num_steps=self.timesteps) if self.encoding else x

        for t in range(self.timesteps):
            spike_data = data[t] if self.encoding else data

            cur1 = self.pool(self.conv1(spike_data))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(self.flatten(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc_out(spk3)
            spk_out, mem_out = self.lif_out(cur4, mem_out)

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
