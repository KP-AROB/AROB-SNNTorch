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

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='cupy')

class SJellyCSNN(BaseJellyNet):
    def __init__(self, 
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)

        self.input_shape = input_shape
        self.n_output = n_output
        self.n_steps = n_steps

        fc_features = int(input_shape[1] / 4)

        self.conv_fc = nn.Sequential(
            layer.Conv2d(self.input_shape[0], 6, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(6),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(6, 16, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(16),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(16 * fc_features**2, 120, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(120, 84, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(84, self.n_output, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

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

        config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.net = self._make_layers(config)
        self.classifier = nn.Sequential(
            nn.Linear(512, self.n_output),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        out = self.net(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_shape[0]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           neuron.IFNode(surrogate_function=surrogate.ATan()),
                        ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class CIFAR10Net(BaseJellyNet):
    def __init__(self, 
                 input_shape=(1, 32, 32),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)
        cfg = [256, 256, 256, 'M', 256, 256, 256, 'M']
        self.layers = self._make_layers(cfg)
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = self.input_shape[0]
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           neuron.IFNode(surrogate_function=surrogate.ATan()),
                        ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr