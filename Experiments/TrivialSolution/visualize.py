import os
import torch
from models.dense_generator import Autoencoder, Encoder
from torch.nn import MSELoss
from utils.datasets.noiseless_dataloader import create_dataloaders_mnist
import numpy as np

import matplotlib.pyplot as plt


def visualize(fp, architecture_params, resume):
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

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

    weight_matmul = np.clip(w2 @ w1, a_min=0, a_max= np.max(w1 @w2))

    plt.imshow(weight_matmul, cmap="hot", interpolation="nearest")
    plt.show()
