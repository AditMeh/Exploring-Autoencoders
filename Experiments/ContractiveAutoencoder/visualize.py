import os
import torch
from models.dense_generator import Autoencoder, Encoder
from torch.nn import MSELoss
from utils.datasets.mnist_dataloaders import create_dataloaders_mnist
import numpy as np

import matplotlib.pyplot as plt


def visualize(fp, architecture_params, resume):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = Autoencoder(**architecture_params).to(device=device)
    if resume:
        autoencoder.load_state_dict(torch.load(os.path.join(fp, "weights/CAE_weights.pt")))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, val_loader = create_dataloaders_mnist(batch_size=4)

    # Sample random datapoint
    x, _ = next(iter(train_loader))
    x = x.to(device=device)

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2, 4)

    for i in range(2):
        axarr[i, 0].imshow(torch.reshape(
            x[2*i], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 1].imshow(torch.reshape(autoencoder(
            x[2*i])[1], torch.Size([28, 28, 1])).detach().cpu().numpy())

        axarr[i, 2].imshow(torch.reshape(
            x[2*i + 1], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 3].imshow(torch.reshape(autoencoder(
            x[2*i + 1])[1], torch.Size([28, 28, 1])).detach().cpu().numpy())

    plt.show()
