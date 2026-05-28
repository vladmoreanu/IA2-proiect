import lighter
import torch


class DnCNN(lighter.Model):
    def __init__(self, channels=3, num_of_layers=17):
        super().__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []

        layers.append(torch.nn.Conv2d(in_channels=channels, out_channels=features,
                                      kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(torch.nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(torch.nn.Conv2d(in_channels=features, out_channels=features,
                                          kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(torch.nn.BatchNorm2d(features))
            layers.append(torch.nn.ReLU(inplace=True))

        layers.append(torch.nn.Conv2d(in_channels=features, out_channels=channels,
                                      kernel_size=kernel_size, padding=padding, bias=False))

        self.dncnn = torch.nn.Sequential(*layers)

    def forward(self, x):
        noise_res = self.dncnn(x)
        return x - noise_res
