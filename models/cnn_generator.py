import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from models.dense_generator import DenseEncoder


class CNNAutoencoder(nn.Module):
    def __init__(self, sizes, h, w, num_dense_layers, fcnn):
        super().__init__()
        self.fcnn = fcnn
        self.encoder = CNNEncoder(sizes)
        if not fcnn:
            self.unflattened_latent_dim = (self.encoder(
                torch.ones((1, sizes[0], w, h))).shape)[1:]
            encoder_input_size = np.product(self.unflattened_latent_dim)

            self.latent_mlp = DenseEncoder(
                [encoder_input_size for _ in range(num_dense_layers)])

        self.decoder = CNNDecoder(sizes)

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
    def __init__(self, sizes):
        super().__init__()
        self.out_seq = nn.Sequential(*[PoolingDownsampleBlock(size_in, size_out) for size_in, size_out
                                       in zip(sizes[0:-1], sizes[1:])])

    def forward(self, x):
        x = self.out_seq(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        sizes = list(reversed(sizes))
        sizes_minus_last = sizes[0:-1]
        self.in_seq = nn.Sequential(*[UnPoolingUpsampleBlock(size_in, size_out, "relu") for size_in, size_out
                                      in zip(sizes_minus_last[0:-1], sizes_minus_last[1:])])

        self.last = UnPoolingUpsampleBlock(
            sizes[-2], sizes[-1], activation="relu")

    def forward(self, x):
        x = self.in_seq(x)
        x = self.last(x)
        return x


class DownsampleBlock(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        # Modify this to create new conv blocks
        # Eg: Throw in pooling, throw in residual connections ... whatever you want
        self.conv_1 = nn.Conv2d(size_in, size_out, 3, padding="valid")
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        return self.act(x)


class UpsampleBlock(nn.Module):
    def __init__(self, size_in, size_out, activation):
        super().__init__()
        # Modify this to create new transpose conv blocks
        # Eg: Throw in dropout, throw in batchnorm ... whatvever you want
        self.up_conv_1 = nn.ConvTranspose2d(size_in, size_out, 3)
        activations = nn.ModuleDict([
            ["relu", nn.ReLU()],
            ["sigmoid", nn.Sigmoid()],
            ["tanh", nn.Tanh()]
        ])
        self.act = activations[activation]

    def forward(self, x):
        x = self.up_conv_1(x)
        return self.act(x)

class PoolingDownsampleBlock(nn.Module):
    def __init__(self, size_in, size_out):
        super().__init__()
        # Modify this to create new conv blocks
        # Eg: Throw in pooling, throw in residual connections ... whatever you want
        self.conv_1 = nn.Conv2d(size_in, size_out, 3, padding="valid")
        self.pool = nn.Conv2d(size_out, size_out, 3, padding="valid")
        #self.pool = nn.MaxPool2d(3, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.conv_1(x)
        x = self.pool(x)
        return self.act(x)

class UnPoolingUpsampleBlock(nn.Module):
    def __init__(self, size_in, size_out, activation):
        super().__init__()
        # Modify this to create new transpose conv blocks
        # Eg: Throw in dropout, throw in batchnorm ... whatvever you want
        self.up_conv_1 = nn.ConvTranspose2d(size_in, size_out, 3)
        self.up_conv_2 = nn.ConvTranspose2d(size_out, size_out, 3)

        activations = nn.ModuleDict([
            ["relu", nn.ReLU()],
            ["sigmoid", nn.Sigmoid()],
            ["tanh", nn.Tanh()]
        ])
        self.act = activations[activation]    
    def forward(self, x):
        x = self.up_conv_1(x)
        x = self.up_conv_2(x)
        return self.act(x)