import torch
from models.autoencoder import Autoencoder, Encoder
from torch.nn import MSELoss
from datasets.noiseless_dataloader import create_dataloaders_mnist
import numpy as np

import matplotlib.pyplot as plt

if __name__ == "__main__":
    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    # Create encoder
    autoencoder = Autoencoder(784, [512, 256], [512], encoder_activation="relu",
                              decoder_activation="relu", final_activation="sigmoid", bias=False).to(device=device)

    autoencoder.load_state_dict(torch.load("no_regularize.pt"))

    # Autoencoder architecture
    print(autoencoder)

    train_loader, val_loader = create_dataloaders_mnist(batch_size=1)

    # Sample random datapoint
    x, _ = next(iter(train_loader))
    x = x.to(device=device)


    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(2,1) 

    _, reconstruction = autoencoder(x)
    axarr[1].imshow(torch.reshape(x, torch.Size([28, 28, 1])).detach().cpu().numpy())
    axarr[0].imshow(torch.reshape(reconstruction, torch.Size([28, 28, 1])).detach().cpu().numpy())


    plt.show()
