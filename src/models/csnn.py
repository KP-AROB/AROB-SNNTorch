import torch
import snntorch as snn
from torch import nn
from .base import BaseFSNN


class S_MNET10(BaseFSNN):
    def __init__(self,
                 input_shape: tuple = (1, 224, 224),
                 n_output: int = 4,
                 n_steps: int = 50,
                 beta: float = 0.95,
                 encoding_type: str = None):
        super().__init__(input_shape, n_output,
                         n_steps, beta, encoding_type)

        k_size = 3
        n_blocks = 4
        fc_size = int(
            (input_shape[1] - n_blocks * k_size + 1 * n_blocks) / 2**n_blocks) - 1

        self.conv1 = nn.Conv2d(input_shape[0], 16, k_size)
        self.lif1 = snn.Leaky(beta=beta)

        self.conv2 = nn.Conv2d(16, 32, k_size)
        self.lif2 = snn.Leaky(beta=beta)

        self.conv3 = nn.Conv2d(32, 32, k_size)
        self.lif3 = snn.Leaky(beta=beta)

        self.conv4 = nn.Conv2d(32, 64, k_size)
        self.lif4 = snn.Leaky(beta=beta)

        self.fc1 = nn.Linear(64 * fc_size**2, 1024)
        self.lif_fc1 = snn.Leaky(beta=beta)

        self.fc_out = nn.Linear(1024, n_output)
        self.lif_out = snn.Leaky(beta=beta, output=True)

        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()
        mem3 = self.lif3.reset_mem()
        mem4 = self.lif4.reset_mem()
        mem_fc1 = self.lif_fc1.reset_mem()
        mem_fc_out = self.lif_out.reset_mem()

        spk_out_rec = []
        mem_out_rec = []

        data = self.encoding(
            x, num_steps=self.n_steps) if self.encoding else x

        for t in range(self.n_steps):
            spike_data = data[t] if self.encoding else data

            cur1 = self.pool(self.conv1(spike_data))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool(self.conv2(spk1))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.pool(self.conv3(spk2))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.pool(self.conv4(spk3))
            spk4, mem4 = self.lif4(cur4, mem4)

            cur5 = self.fc1(self.flatten(spk4))
            spk5, mem_fc1 = self.lif_fc1(cur5, mem_fc1)

            cur_out = self.fc_out(spk5)
            spk_out, mem_fc_out = self.lif_out(cur_out, mem_fc_out)

            spk_out_rec.append(spk_out)
            mem_out_rec.append(mem_fc_out)

        return torch.stack(spk_out_rec, dim=0), torch.stack(mem_out_rec, dim=0)
