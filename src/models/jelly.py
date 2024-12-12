from torch import nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer, encoding
from typing import Tuple


class BaseJellyNet(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int] = (1, 28, 28),
                 n_output: int = 10,
                 n_steps: int = 5,
                 encoding_method: str = 'IF'):
        """Constructor of BaseJellyNet Class

        Args:
            input_shape (Tuple[int], optional): The shape of the input tensor. Defaults to (1, 28, 28).
            n_output (int, optional): The number of output neurons. Defaults to 10.
            n_steps (int, optional): The number of timesteps to encode the input tensor. Defaults to 5.
            encoding_method (str, optional): The neural coding method to perform. Defaults to 'poisson'.
        """
        super().__init__()

        self.input_shape = input_shape
        self.n_output = n_output
        self.n_steps = n_steps
        self.encoding_method = encoding_method
        self.encoder = self.get_encoder()

        functional.set_step_mode(self, step_mode='m')

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

    def get_encoder(self):
        available_methods = ['IF', 'poisson', 'latency', 'phase']
        if self.encoding_method not in available_methods:
            raise ValueError(
                'encoding_method must be of {}'.format(available_methods))
        if self.encoding_method == 'IF':
            return None
        elif self.encoding_method == 'poisson':
            return encoding.PoissonEncoder()
        elif self.encoding_method == 'latency':
            return encoding.LatencyEncoder(T=self.n_steps)
        elif self.encoding_method == 'phase':
            return encoding.WeightedPhaseEncoder(K=self.n_steps)

    def encode_input(self, x):
        if self.encoder:
            x = self.encoder(x)
            if issubclass(type(self.encoder), encoding.StatefulEncoder):
                functional.reset_net(self.encoder)
        return x


class ShallowCSNN(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5,
                 encoding_method='IF'):
        super().__init__(input_shape, n_output, n_steps, encoding_method)

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


class VGG19(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 224, 224),
                 n_output=10,
                 n_steps=5,
                 encoding_method='IF'):
        super().__init__(input_shape, n_output, n_steps, encoding_method)

        cfg = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

        self.net = self.make_layers(cfg, 3)
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(self.feature_map_size, self.n_output),
        )
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x


class SpikingMNET10(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 224, 224),
                 n_output=10,
                 n_steps=5,
                 encoding_method='IF'):
        super().__init__(input_shape, n_output, n_steps, encoding_method)

        k_size = 3
        n_input = input_shape[0]

        self.block1 = nn.Sequential(
            layer.Conv2d(n_input, 16, k_size),
            layer.BatchNorm2d(16),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            layer.Conv2d(16, 32, k_size),
            layer.BatchNorm2d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            layer.Conv2d(32, 64, k_size),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            layer.Conv2d(64, 64, k_size),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.net = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4)

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(12 * 12 * 64, 1024),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5),
            layer.Linear(1024, self.n_output)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x


class SimpleSpikingMNET(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 224, 224),
                 n_output=10,
                 n_steps=5,
                 encoding_method='IF'):
        super().__init__(input_shape, n_output, n_steps, encoding_method)

        k_size = 3
        n_input = input_shape[0]

        self.block1 = nn.Sequential(
            layer.Conv2d(n_input, 16, k_size),
            layer.BatchNorm2d(16),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.block2 = nn.Sequential(
            layer.Conv2d(16, 32, k_size),
            layer.BatchNorm2d(32),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.block3 = nn.Sequential(
            layer.Conv2d(32, 64, k_size),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.block4 = nn.Sequential(
            layer.Conv2d(64, 64, k_size),
            layer.BatchNorm2d(64),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2)
        )

        self.net = nn.Sequential(
            self.block1, self.block2, self.block3, self.block4)

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(12 * 12 * 64, 512),
            neuron.LIFNode(surrogate_function=surrogate.ATan()),
            layer.Linear(512, self.n_output)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x = self.encode_input(x)
        x = self.net(x)
        x = self.classifier(x)
        x = x.mean(0)
        return x
