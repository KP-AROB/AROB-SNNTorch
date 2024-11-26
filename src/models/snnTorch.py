import torch
import snntorch as snn
from torch import nn
from .base import BaseFSNN

class ShallowCSNN(BaseFSNN):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5,
                 beta=0.5,
                 encoding_type=None):
        super().__init__(input_shape, n_output, n_steps, beta, encoding_type)

        n_conv, k_size = 2, 5
        fc_features = ((input_shape[1] - n_conv * k_size + 1 * n_conv) // 2**n_conv) - 1

        self.layers = nn.ModuleList([
            nn.Conv2d(input_shape[0], 6, k_size),
            nn.MaxPool2d(2, 2),
            snn.Leaky(beta=beta),
            nn.Conv2d(6, 16, k_size),
            nn.MaxPool2d(2, 2),
            snn.Leaky(beta=beta),
            nn.Flatten(),
            nn.Linear(16 * fc_features**2, 120),
            snn.Leaky(beta=beta),
            nn.Linear(120, n_output),
            snn.Leaky(beta=beta, output=True)
        ])
    
    def reset_memories(self):
        return [layer.reset_mem() if hasattr(layer, "reset_mem") else None for layer in self.layers]

    def forward(self, x):
        memories = self.reset_memories()

        spk_rec, mem_rec = [], []

        data = self.encoding(x, num_steps=self.n_steps) if self.encoding else x

        for t in range(self.n_steps):
            spike_data = data[t] if self.encoding else data
            memory_idx = 0

            for layer in self.layers:
                if hasattr(layer, "reset_mem"):
                    spike_data, memories[memory_idx] = layer(spike_data, memories[memory_idx])
                    memory_idx += 1
                else:
                    spike_data = layer(spike_data)

            spk_rec.append(spike_data)
            mem_rec.append(memories[-1])

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)
    
