import torchvision.datasets as datasets
from torchvision import transforms
import torch
from torchvision.transforms.transforms import ToTensor, Compos


def create_dataloaders_mnist(batch_size, transforms):
    train_dataset = datasets.MNIST(
        root='./data/', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader
