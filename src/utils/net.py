from torch import nn


def make_vgg_conv_block(cfg, n_input):
    layers = []
    in_channels = n_input

    for x in cfg:
        if x == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        else:
            layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                       nn.BatchNorm2d(x),
                       nn.LeakyReLU()
                       ]
            in_channels = x
    return nn.Sequential(*layers)
