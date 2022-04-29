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

    f, axarr = plt.subplots(1, 2)
    f.set_size_inches(7, 3)

    # Create encoder
    autoencoder = Autoencoder(**architecture_params).to(device=device)
    if resume:
        autoencoder.load_state_dict(torch.load(
            os.path.join(fp, "weights/autoencoder_trivial.pt")))

    # Autoencoder architecture
    print(autoencoder)

    params = [param for param in autoencoder.parameters()]

    w1 = params[0].detach().cpu().numpy()
    w2 = params[1].detach().cpu().numpy()

    # Non-clipped
    axarr[0].imshow(w2 @ w1, cmap="hot", interpolation="nearest")
    axarr[0].title.set_text("No Clipping")

    # Clipped
    axarr[1].title.set_text("With Clipping min(0,x)")
    weight_matmul = np.clip(w2 @ w1, a_min=0, a_max=np.max(w1 @ w2))
    axarr[1].imshow(weight_matmul, cmap="hot", interpolation="nearest")

    plt.show()
