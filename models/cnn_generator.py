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