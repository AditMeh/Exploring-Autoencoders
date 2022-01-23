"""
This experiment shows that the overcomplete autoencoder can learn the trivial solution and achieve 
near perfect performance.
"""

from operator import mod
import torch
import tqdm
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.nn import MSELoss


from models.autoencoder import Autoencoder
from datasets.dataloader import create_dataloaders_mnist
from TorchUtils.training.StatsTracker import StatsTracker


def compute_forward_pass(model, x, optimizer, criterion, update):
    reconstruction = model(x)
    photometric_loss = criterion(reconstruction, x)
    if update:
        model.zero_grad()
        photometric_loss.backward()
        optimizer.step()
    return photometric_loss


def train(model, train_loader, val_loader, device, epochs, lr, batch_size):
    # Initialize autoencoder

    optimizer = Adam(params=model.parameters(), lr=lr)
    statsTracker = StatsTracker()
    criterion = MSELoss(reduction="mean")

    for epoch in range(1, epochs + 1):

        model.train()
        for x, _ in tqdm.tqdm(train_loader):
            x = x.to(device=device)
            photometric_loss = compute_forward_pass(
                model, x, optimizer, criterion, update=True)

            statsTracker.update_curr_losses(photometric_loss.item(), None)

        with torch.no_grad():
            model.eval()
            for x, _ in tqdm.tqdm(val_loader):
                x = x.to(device=device)
                photometric_loss_val = compute_forward_pass(
                    model, x, optimizer, criterion, update=False)

                statsTracker.update_curr_losses(
                    None, photometric_loss_val.item())

        train_loss_epoch = statsTracker.train_loss_curr / \
            (batch_size * len(train_loader))
        val_loss_epoch = statsTracker.val_loss_curr / \
            (batch_size * len(val_loader))

        statsTracker.update_histories(train_loss_epoch, None)

        statsTracker.update_histories(None, val_loss_epoch, model)

        print('Student_network, Epoch {}, Train Loss {}, Val Loss {}'.format(
            epoch, round(train_loss_epoch, 6), round(val_loss_epoch, 6)))


if __name__ == "__main__":
    batch_size = 100
    epochs = 20
    lr = 0.000001

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    train_loader, val_loader = create_dataloaders_mnist(batch_size=batch_size)

    autoencoder = Autoencoder(
        784, [784], [784], final_activation="sigmoid").to(device=device)

    train(autoencoder, train_loader, val_loader,
          device, epochs, lr, batch_size)
