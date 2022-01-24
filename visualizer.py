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
    autoencoder = Autoencoder(
        784, [784], [], final_activation="sigmoid").to(device=device)

    autoencoder.load_state_dict(torch.load("autoencoder.pt"))

    a = torch.rand(size=torch.Size([1, 784])).to(device=device)

    train_loader, val_loader = create_dataloaders_mnist(batch_size=1)

    a, _ = next(iter(train_loader))
    a = a.to(device=device)


    _, recon = autoencoder(a)
    criterion = MSELoss(reduction="sum")

    # f, axarr = plt.subplots(1,2)
    # axarr[0].imshow(torch.reshape(recon, shape= (28, 28)).detach().cpu().numpy())
    # axarr[1].imshow(torch.reshape(a, shape= (28, 28)).detach().cpu().numpy())
    # plt.show()
    print(recon)
    print(a)

    #print((torch.sum(a - recon).item())**2)
    # print(criterion(recon, a))

    params = [(name, param) for name, param in autoencoder.named_parameters()]
    print(params)
    print(len(params))
    w1 = params[0][1].detach().cpu().numpy()
    w2 = params[1][1].detach().cpu().numpy()
    print(w1 @ w2)
    plt.imshow(w1@w2, cmap='winter', interpolation='nearest')
    plt.show()