import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """
    in_size: Size of input, must be a (B, in_size shape)
    enc_sizes: List of hidden layer sizes for the encoder, final value is the latent space size
    dec_sizes: List of hidden layer sizes for the decoder, transform final value in dec_sizes to 
    in_size is applied at the end.
    """

    def __init__(self, in_size, enc_sizes, dec_sizes) -> None:
        super().__init__()
        self.enc_sizes = [in_size, *enc_sizes]
        self.dec_sizes = [enc_sizes[-1], *dec_sizes]

        self.encoder = Encoder(self.enc_sizes)
        self.decoder = Decoder(self.dec_sizes, in_size)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction


class Encoder(nn.Module):
    def __init__(self, enc_sizes) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[FeedforwardLayer(f_in, f_out) for f_in, f_out in zip(enc_sizes, enc_sizes[1:])])

    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_sizes, out_size) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[FeedforwardLayer(f_in, f_out) for f_in, f_out in zip(dec_sizes, dec_sizes[1:])])
        self.last = FeedforwardLayer(dec_sizes[-1], out_size, activation=None)

    def forward(self, x):
        x = self.layers(x)
        x = self.last(x)
        return x


def FeedforwardLayer(f_in, f_out, activation="relu"):
    activations = nn.ModuleDict([
        ["relu", nn.ReLU()]
    ])

    layers = [nn.Linear(f_in, f_out), activations[activation]
              ] if activation is not None else [nn.Linear(f_in, f_out)]

    return nn.Sequential(*layers)
