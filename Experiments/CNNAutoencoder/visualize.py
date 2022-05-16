import os
import torch
from models.cnn_generator import CNNAutoencoder
from models.dense_generator import DenseAutoEncoder
import numpy as np

import matplotlib.pyplot as plt


def visualize(fp, architecture_params, dataloader_params, dataloader_func, resume):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = CNNAutoencoder(**architecture_params).to(device=device)
    if resume:
        autoencoder.load_state_dict(torch.load(
            os.path.join(fp, "weights/cnn_ae.pt")))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, val_loader = dataloader_func(
        **dataloader_params["hyperparams"])

    # Sample random datapoint
    x, _ = next(iter(train_loader))
    x = x.to(device=device)
    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2, 4)

    for i in range(2):
        axarr[i, 0].imshow(torch.permute(x[2*i], (1, 2, 0)).detach().cpu().numpy())
        axarr[i, 1].imshow(torch.permute(
            torch.squeeze(autoencoder(torch.unsqueeze(x[2*i], axis=0))[1]), 
            (1, 2, 0)).detach().cpu().numpy())



        axarr[i, 2].imshow(torch.permute(x[2*i + 1], (1, 2, 0)).detach().cpu().numpy())
        axarr[i, 3].imshow(torch.permute(
            torch.squeeze(autoencoder(torch.unsqueeze(x[2*i + 1], axis=0))[1]), 
            (1, 2, 0)).detach().cpu().numpy())
    plt.show()
