import torch
import snntorch as snn
from torch import nn
from snntorch import surrogate
from snntorch import utils


class CSNN_2C(object):
    def __init__(self,
                 in_shape: tuple = (1, 28, 28),
                 n_class: int = 10,
                 in_features: int = 16,
                 out_features: int = 32,
                 beta: float = 0.5,
                 num_steps: int = 50,
                 k_size: int = 5,
                 device: str = 'cuda'):
        super().__init__()

        self.num_steps = num_steps
        spike_grad = surrogate.fast_sigmoid(slope=25)

        fc_features = int(
            (in_shape[1] - 2 * k_size + 1 * 2) / 4) - 1
        self.net = nn.Sequential(
            nn.Conv2d(in_shape[0], in_features, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Conv2d(in_features, out_features, k_size),
            nn.MaxPool2d(2),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
            nn.Flatten(),
            nn.Linear(out_features*fc_features*fc_features, n_class),
            snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True, output=True)).to(device)

    def forward_pass(self, data):
        mem_rec = []
        spk_rec = []
        utils.reset(self.net)

        for _ in range(self.num_steps):
            spk_out, mem_out = self.net(data)
            spk_rec.append(spk_out)
            mem_rec.append(mem_out)

        return torch.stack(spk_rec), torch.stack(mem_rec)
