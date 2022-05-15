import os
import torch
from models.dense_generator import Autoencoder, Encoder
from torch.nn import MSELoss
from utils.datasets.mnist import DropoutPixelsTransform
import numpy as np

import matplotlib.pyplot as plt


def visualize(fp, architecture_params, dataloader_params, dataloader_func, resume):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = Autoencoder(**architecture_params).to(device=device)
    if resume:
        autoencoder.load_state_dict(torch.load(
            os.path.join(fp, "weights/denoisingae.pt")))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, val_loader = dataloader_func(
        **dataloader_params["hyperparams"])

    dropout_transform = DropoutPixelsTransform(0.5)
    # Sample random datapoint
    target, _ = next(iter(train_loader))
    target = target.to(device=device)
    x = dropout_transform(target)

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2, 6, constrained_layout=True, figsize = [8,2])


    for i in range(2):
        (axarr[i, 0]).title.set_text("Original")
        (axarr[i, 1]).title.set_text("Corrupted")
        (axarr[i, 2]).title.set_text("Reconstructed")
        (axarr[i, 3]).title.set_text("Original")
        (axarr[i, 4]).title.set_text("Corrupted")
        (axarr[i, 5]).title.set_text("Reconstructed")

        axarr[i, 0].imshow(torch.reshape(
            target[2*i], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 1].imshow(torch.reshape(
            x[2*i], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 2].imshow(torch.reshape(autoencoder(
            x[2*i])[1], torch.Size([28, 28, 1])).detach().cpu().numpy())

        axarr[i, 3].imshow(torch.reshape(
            target[2*i + 1], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 4].imshow(torch.reshape(
            x[2*i + 1], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 5].imshow(torch.reshape(autoencoder(
            x[2*i + 1])[1], torch.Size([28, 28, 1])).detach().cpu().numpy())

    
    for i in range(2):
        for j in range(6):
            (axarr[i, j]).set_xticks([])
            (axarr[i, j]).set_yticks([])

    plt.show()
