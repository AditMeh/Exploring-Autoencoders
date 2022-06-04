import os
import torch
from models.cnn_generator import CNNVae
from models.dense_generator import DenseAutoEncoder
import numpy as np

import matplotlib.pyplot as plt
import imageio

NUM_SAMPLES = 50


def visualize(fp, architecture_params, dataloader_params, dataloader_func, resume):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = CNNVae(**architecture_params).to(device=device)
    if resume:
        autoencoder.load_state_dict(torch.load(
            os.path.join(fp, f'weights/cnn_{dataloader_params["name"]}_vae.pt')))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, val_loader = dataloader_func(
        **dataloader_params["hyperparams"])

    # Sample random datapoint
    x = next(iter(train_loader))
    x = x.to(device=device)
    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2, 4)
    for i in range(2):
        for j in range(4):
            (axarr[i][j]).set_xticks([])
            (axarr[i][j]).set_yticks([])

    for i in range(2):
        axarr[i, 0].imshow(torch.permute(
            x[2*i], (1, 2, 0)).detach().cpu().numpy())
        axarr[i, 1].imshow(torch.permute(
            torch.squeeze(autoencoder(torch.unsqueeze(x[2*i], axis=0))[3]),
            (1, 2, 0)).detach().cpu().numpy())

        axarr[i, 2].imshow(torch.permute(
            x[2*i + 1], (1, 2, 0)).detach().cpu().numpy())
        axarr[i, 3].imshow(torch.permute(
            torch.squeeze(autoencoder(torch.unsqueeze(x[2*i + 1], axis=0))[3]),
            (1, 2, 0)).detach().cpu().numpy())

    linear_interpolate(autoencoder, train_loader, device)

    plt.savefig('foo.png')


def linear_interpolate(autoencoder, train_loader, device):
    x = next(iter(train_loader))
    x = x.to(device=device)

    x_1 = torch.unsqueeze(dim=0, input=x[0])
    x_2 = torch.unsqueeze(dim=0, input=x[1])
    latent_1, _, _, _ = autoencoder(x_1)
    latent_2, _, _, _ = autoencoder(x_2)

    latents = torch.stack(
        [latent_1 + (latent_2 - latent_1)*t for t in torch.linspace(0, 1, NUM_SAMPLES)])

    print(latents.shape)
    reconstructions = autoencoder.reconstruct_latent(latents)

    save_images = []
    for img in reconstructions:
        img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        img = (img*255).astype(np.uint8)
        save_images.append(img)

    imageio.mimsave('test.gif', save_images + save_images[::-1])
