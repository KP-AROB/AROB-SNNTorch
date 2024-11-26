import torch
from torch import nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class BaseJellyNet(nn.Module):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__()

        self.input_shape = input_shape
        self.n_output = n_output
        self.n_steps = n_steps


class SJellyCSNN(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)

        self.input_shape = input_shape
        self.n_output = n_output
        self.n_steps = n_steps

        channels = 16
        self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr


class VGGLike(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)

        config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                  'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.net = self._make_layers(config)

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_shape[0]
        for x in cfg:
            if x == 'M':
                layers += [layer.MaxPool2d(2, 2)]
            else:
                layers += [layer.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False),
                           layer.BatchNorm2d(x),
                           neuron.ParametricLIFNode(
                               surrogate_function=surrogate.ATan()),
                           ]
                in_channels = x
        layers += [layer.AvgPool2d(kernel_size=1, stride=1), layer.Flatten()]
        layers += [layer.Linear(512, self.n_output),
                   neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),]
        return nn.Sequential(*layers)

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        out = self.net(x_seq)
        out_fr = out.mean(0)
        return out_fr
