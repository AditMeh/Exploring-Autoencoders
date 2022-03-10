"""
This experiment is for the contrastive autoencoderr
"""

import json
from operator import mod
import os
import torch
import tqdm
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import MSELoss


from models.dense_generator import VariationalAutoencoder
from utils.datasets.noiseless_dataloader import create_dataloaders_mnist
from utils.TorchUtils.training.StatsTracker import StatsTracker


def ELBO(x, mu, logvar, reconstruction):
    # = -log(sigma) + 1/2 (sigma^2 + mu^2) - 1/2

    # Gaussian log likelihood is proportional to -ve MSE,
    
    KLD_vector = -0.5*(1 + logvar - torch.exp(logvar) - torch.pow(mu, 2))
    
    KLD_scalar = torch.sum(torch.sum(KLD_vector, axis=1), axis = 0)
    

    gaussian_log_likelihood = (-1) * \
        MSELoss(reduction="sum")(reconstruction, x)

    # ELBO is defined as -KL divergence + log likelihood
    # Therefore since we want to maximize the ELBO, we equivalently need to minimize KLD - log likelihood
    return KLD_scalar - gaussian_log_likelihood


def compute_forward_pass(model, x, optimizer, criterion, update):
    latent, mean, logvar, reconstruction = model(x)
    VAE_loss = criterion(x, mean, logvar, reconstruction)
    if update:
        model.zero_grad()
        VAE_loss.backward()
        optimizer.step()
    return VAE_loss


def train(model, train_loader, val_loader, device, epochs, lr, batch_size, weight):
    # Initialize autoencoder

    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, min_lr=0.00001, verbose=True)

    statsTracker = StatsTracker(
        batch_size * len(train_loader), batch_size * len(val_loader))

    criterion = ELBO

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
                photometric_loss_val = compute_forward_pass(model, x)

                statsTracker.update_curr_losses(
                    None, photometric_loss_val.item())

        train_loss_epoch, val_loss_epoch = statsTracker.compute_means()
        assert((statsTracker.train_loss_curr /
               (batch_size * len(train_loader))) == train_loss_epoch)
        assert((statsTracker.val_loss_curr /
               (batch_size * len(val_loader))) == val_loss_epoch)

        statsTracker.update_histories(train_loss_epoch, None)

        statsTracker.update_histories(None, val_loss_epoch, model)

        scheduler.step(val_loss_epoch)
        print('Student_network, Epoch {}, Train Loss {}, Val Loss {}'.format(
            epoch, round(train_loss_epoch, 6), round(val_loss_epoch, 6)))

        statsTracker.reset()

    return statsTracker.best_model


def run_experiment(fp, training_params, architecture_params, resume):
    batch_size = training_params["batch_size"]
    epochs = training_params["epochs"]
    lr = training_params["lr"]
    weight = training_params["weight"]

    device = (torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    train_loader, val_loader = create_dataloaders_mnist(batch_size=batch_size)

    autoencoder = VariationalAutoencoder(
        **(architecture_params)).to(device=device)

    if resume:
        autoencoder.load_state_dict(torch.load(
            os.path.join(fp, "weights/VAE_weights.pt")))

    print(autoencoder)
    best_model = train(autoencoder, train_loader, val_loader,
                       device, epochs, lr, batch_size, weight)
    torch.save(best_model, os.path.join(fp, "weights/VAE_weights.pt"))
