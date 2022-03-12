import os
import torch
from models.dense_generator import VariationalAutoencoder
from torch.nn import MSELoss
from utils.datasets.noiseless_dataloader import create_dataloaders_mnist
import numpy as np
import tqdm

import matplotlib.pyplot as plt

def plot_latent(train_loader, autoencoder, device):
    f_1, axarr_1 = plt.subplots()

    for (x, y) in tqdm.tqdm(train_loader):
        z, _, _,_ = autoencoder(x.to(device))
        z = z.detach().cpu().numpy()
        pos = axarr_1.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

    f_1.colorbar(pos)


def plot_samples(autoencoder, device):
    f_2, axarr_2 = plt.subplots()
    NUM_IMAGES = 12
    img = np.zeros((NUM_IMAGES*28, NUM_IMAGES*28))


    for i, x in enumerate(np.linspace(-1, 1, NUM_IMAGES)):
        for j, y in enumerate(np.linspace(-1, 1, NUM_IMAGES)):
            z = torch.Tensor([[x, y]]).to(device)
            reconstruction = autoencoder.decoder(z)
            reconstruction = torch.reshape(reconstruction, shape=torch.Size([28 , 28]))
            reconstruction = reconstruction.detach().cpu().numpy()

            img[i*28:(i+1)*28, j*28:(j+1)*28] = reconstruction
    axarr_2.imshow(img)


def visualize(fp, architecture_params, resume):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = VariationalAutoencoder(**architecture_params).to(device=device)
    if resume:
        autoencoder.load_state_dict(torch.load(os.path.join(fp, "weights/VAE_weights.pt")))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, _ = create_dataloaders_mnist(batch_size=32)

    # Sample random datapoint
    x, _ = next(iter(train_loader))
    x = x.to(device=device)

    # generate image comparison subplot
    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2, 4)

    for i in range(2):
        axarr[i, 0].imshow(torch.reshape(
            x[2*i], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 1].imshow(torch.reshape(autoencoder(
            x[2*i])[3], torch.Size([28, 28, 1])).detach().cpu().numpy())

        axarr[i, 2].imshow(torch.reshape(
            x[2*i + 1], torch.Size([28, 28, 1])).detach().cpu().numpy())
        axarr[i, 3].imshow(torch.reshape(autoencoder(
            x[2*i + 1])[3], torch.Size([28, 28, 1])).detach().cpu().numpy())


    # # Latent space visualization
    plot_latent(train_loader, autoencoder, device)

    # sample latents -> image
    # generate image comparison subplot
    plot_samples(autoencoder, device)

    plt.show()

