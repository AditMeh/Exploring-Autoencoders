import torch.nn as nn
import torch.nn.functional as F


# AUTOENCODER ARCHITECTURE
class Network(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.encoder = Encoder(sizes)
        self.decoder = Decoder(sizes)
    def forward(self, x):
        latent = self.encoder(x)
        return latent, self.decoder(latent)

class Encoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.out_seq = nn.Sequential(*[DownsampleBlock(size_in, size_out) for size_in, size_out 
                                      in zip(sizes[0:-1], sizes[1:])])
    def forward(self, x):
        x = self.out_seq(x)
        return x

class Decoder(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        sizes = list(reversed(sizes))
        self.in_seq = nn.Sequential(*[UpsampleBlock(size_in, size_out) for size_in, size_out 
                                      in zip(sizes[0:-1], sizes[1:])])
    def forward(self, x):
        x = self.in_seq(x)
        return x

# CONV BLOCK
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
    def __init__(self, size_in, size_out):
        super().__init__()
        # Modify this to create new transpose conv blocks
        # Eg: Throw in dropout, throw in batchnorm ... whatvever you want
        self.up_conv_1 = nn.ConvTranspose2d(size_in, size_out, 3)
        self.act = nn.ReLU()
    def forward(self, x):
        x = self.up_conv_1(x)
        return self.act(x)