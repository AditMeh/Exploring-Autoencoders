from enum import auto
import os
from re import L
import torch
from models.dense_generator import VariationalAutoencoder
from utils.datasets.noiseless_dataloader import create_dataloaders_mnist
import numpy as np
import tqdm
import imageio

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

def generate_gif(autoencoder, device):
    NUM_SAMPLES = 30
    SAVE_IMAGE_W = 10
    fill_image = np.zeros((NUM_SAMPLES ,SAVE_IMAGE_W*28, SAVE_IMAGE_W*28))
    
    train_iter = iter(create_dataloaders_mnist(batch_size=SAVE_IMAGE_W**2)[0])
    x_1, x_2 = (next(train_iter)[0]).to(device), (next(train_iter)[0]).to(device)
    z_1, _, _,_ = autoencoder(x_1)
    z_2, _, _,_ = autoencoder(x_2)

    zs = torch.stack([z_1 + (z_2 - z_1)*t for t in torch.linspace(0, 1, NUM_SAMPLES)])

    for i in range(NUM_SAMPLES):
        batch_decoded = autoencoder.decoder(zs[i]).reshape(SAVE_IMAGE_W,SAVE_IMAGE_W,28,28).detach().cpu().numpy()

        for j in range(SAVE_IMAGE_W):
            for k in range(SAVE_IMAGE_W):
                fill_image[i, j*28:(j+1)*28, k*28:(k+1)*28] = batch_decoded[j, k, :, :]

    images = (fill_image*255).astype(np.uint8)

    save_images = []
    for img in images:
        save_images.append(img)
    imageio.mimsave('test.gif', save_images + save_images[::-1])

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

    # Interpolation
    generate_gif(autoencoder, device)

    plt.show()

