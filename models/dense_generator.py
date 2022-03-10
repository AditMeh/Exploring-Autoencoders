from re import M
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    """
    in_size: Size of input, must be a (B, in_size shape)
    enc_sizes: List of hidden layer sizes for the encoder, final value is the latent space size
    dec_sizes: List of hidden layer sizes for the decoder, transform final value in dec_sizes to 
    in_size is applied at the end.
    """

    def __init__(self, in_size, enc_sizes, dec_sizes, z_dim, 
    encoder_activation=None, decoder_activation=None, final_activation=None, bias=True) -> None:
        super().__init__()
        self.enc_sizes = [in_size, *enc_sizes]
        self.dec_sizes = [z_dim, *dec_sizes]

        self.encoder = Encoder(self.enc_sizes, bias, encoder_activation)

        self.mean_encoder = nn.Linear(enc_sizes[-1], z_dim)
        self.variance_encoder = nn.Linear(enc_sizes[-1], z_dim)

        self.decoder = Decoder(self.dec_sizes, in_size,
                               bias, decoder_activation, final_activation)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mean

    def forward(self, x):
        latent = self.encoder(x)
        mean, logvar = self.mean_encoder(latent), self.variance_encoder(latent)

        z = self.reparameterize(mean, logvar)
        reconstruction = self.decoder(z)
        return z, mean, logvar, reconstruction
    


class Autoencoder(nn.Module):
    """
    in_size: Size of input, must be a (B, in_size shape)
    enc_sizes: List of hidden layer sizes for the encoder, final value is the latent space size
    dec_sizes: List of hidden layer sizes for the decoder, transform final value in dec_sizes to 
    in_size is applied at the end.
    """

    def __init__(self, in_size, enc_sizes, dec_sizes, encoder_activation=None, decoder_activation=None, final_activation=None, bias=True) -> None:
        super().__init__()
        self.enc_sizes = [in_size, *enc_sizes]
        self.dec_sizes = [enc_sizes[-1], *dec_sizes]

        self.encoder = Encoder(self.enc_sizes, bias, encoder_activation)
        self.decoder = Decoder(self.dec_sizes, in_size,
                               bias, decoder_activation, final_activation)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return latent, reconstruction


class Encoder(nn.Module):
    def __init__(self, enc_sizes, bias=True, activation=None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[FeedforwardLayer(f_in, f_out, bias, activation) for f_in, f_out in zip(enc_sizes, enc_sizes[1:])])

    def forward(self, x):
        x = self.layers(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dec_sizes, out_size, bias=True, decoder_activation=None, final_activation=None) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            *[FeedforwardLayer(f_in, f_out, activation=decoder_activation, bias=bias) for f_in, f_out in zip(dec_sizes, dec_sizes[1:])])
        self.last = FeedforwardLayer(
            dec_sizes[-1], out_size, activation=final_activation, bias=bias)

    def forward(self, x):
        x = self.layers(x)
        x = self.last(x)
        return x


def FeedforwardLayer(f_in, f_out, bias=True, activation="relu"):
    activations = nn.ModuleDict([
        ["relu", nn.ReLU()],
        ["sigmoid", nn.Sigmoid()],
        ["tanh", nn.Tanh()]
    ])

    layers = [nn.Linear(f_in, f_out, bias=bias), activations[activation]
              ] if activation is not None else [nn.Linear(f_in, f_out, bias=bias)]

    return nn.Sequential(*layers)
