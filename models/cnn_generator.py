import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.dense_generator import DenseEncoder
import math


def conv_func(p, d, k, s, h):
    return math.floor(((h + 2*p - d * (k-1) - 1)/s) + 1)\



class CNNAutoencoder(nn.Module):
    def __init__(self, sizes, h, w, num_dense_layers, fcnn):
        super().__init__()
        self.fcnn = fcnn
        self.encoder = CNNEncoder(sizes, h)
        if not fcnn:
            self.unflattened_latent_dim = (self.encoder(
                torch.ones((1, sizes[0], w, h))).shape)[1:]
            encoder_input_size = np.product(self.unflattened_latent_dim)

            self.latent_mlp = DenseEncoder(
                [encoder_input_size for _ in range(num_dense_layers)])

        self.decoder = CNNDecoder(sizes, self.encoder.output_padding_flags)

    def forward(self, x):
        latent = self.encoder(x)
        if not self.fcnn:

            latent = torch.flatten(latent, start_dim=1)

            dense_encoder = self.latent_mlp(latent)

            reconstruction = self.decoder(torch.reshape(
                dense_encoder, [dense_encoder.shape[0], *self.unflattened_latent_dim]))
        else:
            reconstruction = self.decoder(latent)

        return latent, reconstruction


class CNNEncoder(nn.Module):
    def __init__(self, sizes, img_dim):
        super().__init__()

        modules_list = []
        self.output_padding_flags = []
        for size_in, size_out in zip(sizes[0:-1], sizes[1:]):
            modules_list.append(
                nn.Conv2d(size_in, size_out, kernel_size=3, stride=2, padding=1))

            if (img_dim % 2) == 0:  # even img dim
                self.output_padding_flags.append(1)
            else:
                self.output_padding_flags.append(0)

            img_dim = conv_func(1, 1, 3, 2, img_dim)

        self.out_seq = nn.Sequential(*modules_list)

    def forward(self, x):
        x = self.out_seq(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, sizes, padding_flags):
        super().__init__()

        self.in_seq = self.create_sequential(sizes, padding_flags)

        if padding_flags[0]:
            self.last = UpsampleBlock(
                sizes[1], sizes[0], 1, activation="sigmoid")
        else:
            self.last = UpsampleBlock(
                sizes[1], sizes[0], 0, activation="sigmoid")

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
            if padding_flags[i]:
                module_list.append(UpsampleBlock(size_in, size_out, 1, "relu"))
            else:
                module_list.append(UpsampleBlock(size_in, size_out, 0, "relu"))
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

if __name__ == "__main__":
    for cap in range(3, 8, 1):
        for range_cap in range(5, 15, 1):
            for img_size in range(70, 150, range_cap):

                model = CNNAutoencoder(
                    [3 for _ in range(cap)],
                    img_size,
                    img_size,
                    2,
                    fcnn=True
                )

                a = torch.ones((1, 3, img_size, img_size))
                print(a.shape)
                print(model(a)[1].shape)
                print("----------------")

                assert model(a)[1].shape == a.shape

