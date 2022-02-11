"""
This experiment shows that the overcomplete autoencoder can learn the trivial solution and achieve 
near perfect performance.
"""

import json
import os
import torch
import tqdm
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


from models.autoencoder import Autoencoder, Encoder
from utils.datasets.noiseless_dataloader import create_dataloaders_mnist
from utils.TorchUtils.training.StatsTracker import StatsTracker


def compute_forward_pass(model, x, optimizer, criterion, update):
    latent, reconstruction = model(x)
    photometric_loss = criterion(reconstruction, x)
    if update:
        model.zero_grad()
        photometric_loss.backward()
        optimizer.step()
    return photometric_loss


def train(model, train_loader, val_loader, device, epochs, lr, batch_size):
    # Initialize autoencoder

    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, min_lr=0.00001, verbose=True)

    statsTracker = StatsTracker(
        batch_size * len(train_loader), batch_size * len(val_loader))
    criterion = MSELoss(reduction="sum")

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

        train_loss_epoch, val_loss_epoch = statsTracker.compute_means()

        statsTracker.update_histories(train_loss_epoch, None)

        statsTracker.update_histories(None, val_loss_epoch, model)

        scheduler.step(val_loss_epoch)
        print('Student_network, Epoch {}, Train Loss {}, Val Loss {}'.format(
            epoch, round(train_loss_epoch, 6), round(val_loss_epoch, 6)))

        statsTracker.reset()

    return statsTracker.best_model


def run_experiment(fp, training_params, architecture_params):
    batch_size = training_params["batch_size"]
    epochs = training_params["epochs"]
    lr = training_params["lr"]

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    train_loader, val_loader = create_dataloaders_mnist(batch_size=batch_size)

    autoencoder = Autoencoder(**(architecture_params)).to(device=device)
    autoencoder.load_state_dict(torch.load(
        os.path.join(fp, "weights/autoencoder_trivial.pt")))

    print(autoencoder)
    best_model = train(autoencoder, train_loader, val_loader,
                       device, epochs, lr, batch_size)
    torch.save(best_model, os.path.join(
        fp, 'weights/autoencoder_trivial.pt'))
