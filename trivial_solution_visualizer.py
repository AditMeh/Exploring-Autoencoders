from turtle import shape
import torch
from models.autoencoder import Autoencoder, Encoder
from torch.nn import MSELoss
from datasets.dataloader import create_dataloaders_mnist
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = Autoencoder(784, [784], [], encoder_activation=None,
                              decoder_activation=None, final_activation=None, bias=False).to(device=device)

    autoencoder.load_state_dict(torch.load("autoencoder_trivial.pt"))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, val_loader = create_dataloaders_mnist(batch_size=1)

    # Sample random datapoint
    x, _ = next(iter(train_loader))
    x = x.to(device=device)

    _, reconstruction = autoencoder(x)

    params = [param for param in autoencoder.parameters()]

    w1 = params[0].detach().cpu().numpy()
    w2 = params[1].detach().cpu().numpy()

    weight_matmul = w2 @ w1

    plt.imshow(weight_matmul, cmap="hot", interpolation="nearest")
    plt.show()
