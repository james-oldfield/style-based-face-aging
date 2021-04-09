import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    """Generator network."""
    def __init__(self, conv_dim=64, repeat_num=6):
        super(Encoder, self).__init__()

        self.conv_layers = nn.ModuleList([nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3)])
        self.IN_layers = nn.ModuleList([nn.InstanceNorm2d(conv_dim)])

        maps = [64, 128, 256, 512, 512, 512, 512]

        # Down-sampling layers.
        for i in range(6):
            self.conv_layers.append(nn.Conv2d(maps[i], maps[i + 1], kernel_size=4, stride=2, padding=1))
            self.IN_layers.append(nn.InstanceNorm2d(maps[i + 1]))

    def forward(self, x):
        activations = []

        for layer_i, (conv, IN) in enumerate(zip(self.conv_layers, self.IN_layers)):
            x = conv(x)

            activations.append(x)
            x = F.relu(IN(x))

        return activations


class Decoder(nn.Module):
    def __init__(self, num_layers_to_skip):
        super(Decoder, self).__init__()

        self.num_layers_to_skip = num_layers_to_skip

        self.conv_layers = nn.ModuleList([nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)])
        self.IN_layers = nn.ModuleList([nn.InstanceNorm2d(512)])
        self.mu_AT_layers = nn.ModuleList([nn.Linear(512, 512)])
        self.sigma_AT_layers = nn.ModuleList([nn.Linear(512, 512)])

        self.in0 = nn.InstanceNorm2d(512)

        maps = [512, 512, 512, 512, 256, 128, 64]

        # Up-sampling layers.
        for i in range(6):
            self.conv_layers.append(nn.ConvTranspose2d(maps[i], maps[i + 1], kernel_size=4, stride=2, padding=1))
            self.IN_layers.append(nn.InstanceNorm2d(maps[i + 1]))

            # affine transformations of the 1st/2nd clfs' moments, separately.
            self.mu_AT_layers.append(nn.Linear(maps[i + 1], maps[i + 1]))
            self.sigma_AT_layers.append(nn.Linear(maps[i + 1], maps[i + 1]))

        self.ConvOut = nn.Conv2d(maps[-1], 3, kernel_size=7, stride=1, padding=3)

    def forward(self, layers, z):
        x = F.relu(self.in0(layers[-1]))

        for layer_i, (conv, IN, fcMu, fcSigma) in enumerate(zip(self.conv_layers, self.IN_layers, self.mu_AT_layers, self.sigma_AT_layers)):
            x = conv(x)

            mu = fcMu(z[-(layer_i + 1)][0])[:, :, None, None]
            sigma = fcSigma(z[-(layer_i + 1)][1])[:, :, None, None]

            # optionally omit skip connection(s)
            if layer_i < len(self.conv_layers) - self.num_layers_to_skip:
                x += layers[-(layer_i + 1)]

            x = sigma * IN(x) + mu
            x = F.relu(x)

        x = self.ConvOut(x)

        return x


class Classifier(nn.Module):
    def __init__(self, num_classes, conv_dim=64, n_flat=2 * 2 * 512):
        "n_flat: number of dimensions flattened (last conv layer)"
        super(Classifier, self).__init__()
        self.n_flat = n_flat
        self.num_classes = num_classes

        self.layers = nn.ModuleList()

        self.layers.append(nn.Conv2d(3, conv_dim, kernel_size=7, stride=1, padding=3))
        self.layers.append(nn.Conv2d(conv_dim * 1, conv_dim * 2, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Conv2d(conv_dim * 2, conv_dim * 4, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Conv2d(conv_dim * 4, conv_dim * 8, kernel_size=4, stride=2, padding=1))

        self.layers.append(nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=4, stride=2, padding=1))
        self.layers.append(nn.Conv2d(conv_dim * 8, conv_dim * 8, kernel_size=4, stride=2, padding=1))

        self.fc1 = nn.Linear(self.n_flat, self.num_classes)

    def forward(self, x, y=None):
        activations = []

        # return list of activations to use in the generator
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
            activations.append(x)

        x = x.view(-1, self.n_flat)
        x = self.fc1(x)
        activations.append(x)

        # get the elements of the logits of target class(es) only
        logits = x if y is None else x.gather(1, y.view(-1, 1))

        return activations, logits
