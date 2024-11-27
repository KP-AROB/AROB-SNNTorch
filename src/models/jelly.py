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
        fc_size = int(input_shape[1] / 2**4)
 
        self.conv_fc = nn.Sequential(
            layer.Conv2d(self.input_shape[0], 64, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 64, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(64),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2), 

            layer.Conv2d(64, 128, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(128, 128, kernel_size=3,
                         padding=1, bias=False),
            layer.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2), 

            layer.Flatten(),
            layer.Linear(128 * fc_size * fc_size, 256, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(256, self.n_output, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x: torch.Tensor):
        x_seq = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        x_seq = self.conv_fc(x_seq)
        fr = x_seq.mean(0)
        return fr

class S_VGG(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)
        config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256,
                  'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        
        self.net = self._make_layers(config)
        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(4096, self.n_output),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan())
        )

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
        layers += [layer.Flatten()]
        return nn.Sequential(*layers)

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        out = self.net(x_seq)
        out = self.dense(out)
        out_fr = out.mean(0)
        return out_fr

class MNET_Jelly(BaseJellyNet):
    def __init__(self,
                 input_shape=(1, 28, 28),
                 n_output=10,
                 n_steps=5):
        super().__init__(input_shape, n_output, n_steps)
        cfg = [16, 'M', 32, 'M', 32, 'M', 64, 'M']
        
        self.net = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            layer.Linear(self.feature_map_size, 1024),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5),
            layer.Linear(1024, self.n_output),
            neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
        )

        functional.set_step_mode(self, step_mode='m')
        

    def _make_layers(self, cfg):
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
                    layer.Conv2d(in_channels, x, kernel_size=5, bias=False),
                    layer.BatchNorm2d(x),
                    neuron.ParametricLIFNode(surrogate_function=surrogate.ATan()),
                ]
                in_channels = x
                height -= 4
                width -= 4

        if height <= 0 or width <= 0:
            raise ValueError("Feature map size became non-positive. Check input shape or layer configuration.")

        layers += [layer.Flatten()]
        self.feature_map_size = height * width * in_channels 
        return nn.Sequential(*layers)

    def forward(self, x):
        x_seq = x.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1)
        out = self.net(x_seq)
        out = self.classifier(out)
        # average decision accross timesteps
        out_fr = out.mean(0)
        return out_fr