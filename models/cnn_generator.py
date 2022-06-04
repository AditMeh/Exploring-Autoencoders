import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.dense_generator import DenseEncoder, FeedforwardLayer
import math


def conv_func(p, d, k, s, h):
    return math.floor(((h + 2*p - d * (k-1) - 1)/s) + 1)


class CNNVae(nn.Module):
    def __init__(self, sizes, h, w, num_dense_layers, z_dim):
        super().__init__()
        self.encoder = CNNEncoder(sizes, h, w)
        self.unflattened_latent_dim = (self.encoder(
            torch.ones((1, sizes[0], h, w))).shape)[1:]
        encoder_input_size = np.product(self.unflattened_latent_dim)

        self.latent_mlp = DenseEncoder(
            [encoder_input_size for _ in range(num_dense_layers)])

        self.mu = FeedforwardLayer(encoder_input_size, z_dim)
        self.logvar = FeedforwardLayer(encoder_input_size, z_dim)
        self.unproject = FeedforwardLayer(z_dim, encoder_input_size)

        self.decoder = CNNDecoder(sizes, self.encoder.output_padding_flags)

    def forward(self, x):
        pre_latent = self.encoder(x)
        pre_latent = torch.flatten(pre_latent, start_dim=1)
        pre_latent = self.latent_mlp(pre_latent)

        mu, logvar = self.mu(pre_latent), self.logvar(pre_latent)

        z = self.reparameterize(mu, logvar)

        unproject = self.unproject(z)
        reconstruction = self.decoder(torch.reshape(
            unproject, [unproject.shape[0], *self.unflattened_latent_dim]))

        return z, mu, logvar, reconstruction

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mean

    def reconstruct_latent(self, z):
        unproject = self.unproject(z)
        reconstruction = self.decoder(torch.reshape(
            unproject, [unproject.shape[0], *self.unflattened_latent_dim]))
        return reconstruction


class CNNAutoencoder(nn.Module):
    def __init__(self, sizes, h, w, num_dense_layers, fcnn):
        super().__init__()
        self.fcnn = fcnn
        self.encoder = CNNEncoder(sizes, h, w)
        if not fcnn:
            self.unflattened_latent_dim = (self.encoder(
                torch.ones((1, sizes[0], h, w))).shape)[1:]
            encoder_input_size = np.product(self.unflattened_latent_dim)

            self.latent_mlp = DenseEncoder(
                [encoder_input_size for _ in range(num_dense_layers)])

        self.decoder = CNNDecoder(sizes, self.encoder.output_padding_flags)

    def forward(self, x):
        latent = self.encoder(x)
        if not self.fcnn:

            latent = torch.flatten(latent, start_dim=1)

            latent = self.latent_mlp(latent)

            reconstruction = self.decoder(torch.reshape(
                latent, [latent.shape[0], *self.unflattened_latent_dim]))
        else:
            reconstruction = self.decoder(latent)

        return latent, reconstruction


class CNNEncoder(nn.Module):
    def __init__(self, sizes, h, w):
        super().__init__()

        modules_list = []
        self.output_padding_flags = []
        for size_in, size_out in zip(sizes[0:-1], sizes[1:]):
            modules_list.append(DownsampleBlock(size_in, size_out))

            self.output_padding_flags.append(
                (int(h % 2 == 0), int(w % 2 == 0)))

            h, w = conv_func(1, 1, 3, 2, h), conv_func(1, 1, 3, 2, w)

        self.out_seq = nn.Sequential(*modules_list)

    def forward(self, x):
        x = self.out_seq(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, sizes, padding_flags):
        super().__init__()

        self.in_seq = self.create_sequential(sizes, padding_flags)

        self.last = UpsampleBlock(
            sizes[1], sizes[0], padding_flags[0], activation="sigmoid")

    def forward(self, x):
        x = self.in_seq(x)
        x = self.last(x)
        return x

    @staticmethod
    def create_sequential(sizes, padding_flags):
        module_list = []
        sizes = list(reversed(sizes))
        padding_flags = padding_flags[::-1]
        sizes_minus_last = sizes[0:-1]

        for i, size_tup in enumerate(zip(sizes_minus_last[0:-1], sizes_minus_last[1:])):
            size_in, size_out = size_tup
            module_list.append(UpsampleBlock(
                size_in, size_out, padding_flags[i], "relu"))
        return nn.Sequential(*module_list)


class DownsampleBlock(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        # Modify this to create new conv blocks
        # Eg: Throw in pooling, throw in residual connections ... whatever you want
        self.conv_1 = nn.Conv2d(
            size_in, size_out, kernel_size=3, stride=2, padding=1)
        self.bn_1 = nn.BatchNorm2d(size_out)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn_1(x)
        return self.act(x)


class UpsampleBlock(nn.Module):
    def __init__(self, size_in, size_out, output_padding, activation):
        super().__init__()
        # Modify this to create new transpose conv blocks
        # Eg: Throw in dropout, throw in batchnorm ... whatvever you want
        self.up_conv_1 = nn.ConvTranspose2d(
            size_in, size_out, kernel_size=3, stride=2, padding=1, output_padding=output_padding)
        activations = nn.ModuleDict([
            ["relu", nn.ReLU()],
            ["sigmoid", nn.Sigmoid()],
            ["tanh", nn.Tanh()]
        ])
        self.bn_1 = nn.BatchNorm2d(size_out)

        self.act = activations[activation]

    def forward(self, x):
        x = self.up_conv_1(x)
        x = self.bn_1(x)
        return self.act(x)
