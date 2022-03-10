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


from models.dense_generator import Autoencoder
from utils.datasets.noiseless_dataloader import create_dataloaders_mnist
from utils.TorchUtils.training.StatsTracker import StatsTracker


"""
The below code was adapted from 
https://stackoverflow.com/questions/58249160/how-to-implement-contractive-autoencoder-in-pytorch
"""
def compute_forward_pass(model, x, optimizer, weight, device, update):
    # Flip on the grad switches for the GT tensor
    x.requires_grad_(True)
    x.retain_grad()
    latent, reconstruction = model(x)

    reconstruction_loss = MSELoss(reduction="sum")(reconstruction, x)

    # Backprop from the latent to the leaves (now including the GT tensor). Retain graph, as we need to backprop through
    # the latent once more for the MSE term. We pass in downstream gradient of ones, because we only want a gradient
    # of dz/dx, so dot producting dy/dz set to a vector of ones with dz/dx returns just dz/dx
    latent.backward(torch.ones(latent.size()).to(device), retain_graph=True)
    
    loss2 = torch.sqrt(torch.sum(torch.pow(x.grad, 2))) # Comptue the frobenius norm on the gradients
    x.grad.data.zero_()
    loss = reconstruction_loss + (weight*loss2)
    x.requires_grad_(False)
    if update:
        model.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

def compute_mse(model, x):
    latent, reconstruction = model(x)
    photometric_loss = MSELoss(reduction="sum")(reconstruction, x)
    return photometric_loss

def train(model, train_loader, val_loader, device, epochs, lr, batch_size, weight):
    # Initialize autoencoder

    optimizer = Adam(params=model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', factor=0.1, patience=3, min_lr=0.00001, verbose=True)

    statsTracker = StatsTracker(
        batch_size * len(train_loader), batch_size * len(val_loader))

    for epoch in range(1, epochs + 1):

        model.train()
        for x, _ in tqdm.tqdm(train_loader):
            x = x.to(device=device)
            photometric_loss = compute_forward_pass(
                model, x, optimizer, weight, device, update=True)
            statsTracker.update_curr_losses(photometric_loss.item(), None)
        
        with torch.no_grad():
            model.eval()
            for x, _ in tqdm.tqdm(val_loader):
                x = x.to(device=device)
                photometric_loss_val = compute_mse(model, x)

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

    autoencoder = Autoencoder(**(architecture_params)).to(device=device)

    if resume:
        autoencoder.load_state_dict(torch.load(
            os.path.join(fp, "weights/CAE_weights.pt")))

    print(autoencoder)
    best_model = train(autoencoder, train_loader, val_loader,
                       device, epochs, lr, batch_size, weight)
    torch.save(best_model, os.path.join(fp, "weights/CAE_weights.pt"))
