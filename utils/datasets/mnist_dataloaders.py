import torchvision.datasets as datasets
import torch
from torchvision.transforms.transforms import ToTensor, Compose, Lambda
from torch.nn.functional import dropout

class DropoutPixelsTransform:
    def __init__(self, prob) -> None:
        self.prob = prob
    def __call__(self, tensor):
        return dropout(tensor, self.prob) # inplace dropout

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

def create_dataloaders_mnist(batch_size, tanh_normalize=False):

    transforms = [ToTensor(), Lambda(lambda x: torch.flatten(x))]

    if tanh_normalize:
        transforms.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))

    transforms = Compose(transforms)
    train_dataset = datasets.MNIST(
        root='./data/', train=True, download=True, transform=transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = datasets.MNIST(
        root='./data/', train=False, download=True, transform=transforms)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
