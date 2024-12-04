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

    def make_layers(self, cfg, k_size=5):
        layers = []
        in_channels = self.input_shape[0]
        height, width = self.input_shape[1], self.input_shape[2]

        for x in cfg:
            if x == 'M':
                layers += [layer.MaxPool2d(2, 2)]
                height //= 2
                width //= 2
            else:
                layers += [
                    layer.Conv2d(in_channels, x,
                                 kernel_size=k_size, bias=False),
                    layer.BatchNorm2d(x),
                    neuron.IFNode(surrogate_function=surrogate.ATan()),
                ]
                in_channels = x
                height -= k_size - 1
                width -= k_size - 1

        if height <= 0 or width <= 0:
            raise ValueError(
                "Feature map size became non-positive. Check input shape or layer configuration.")
        self.feature_map_size = height * width * in_channels
        return nn.Sequential(*layers)


class ShallowCSNN(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)

        cfg = [16, 'M', 32, 'M', 64, 'M']
        self.net = self.make_layers(cfg, 3)
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(self.feature_map_size, 256),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(256, self.n_output, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x
