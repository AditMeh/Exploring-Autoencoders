import torch
import torch.nn as nn
import numpy as np

import os
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


CIFAR_CLASSES = ['plane', 'car', 'bird', 'cat',
                 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CIFARDataset(torch.utils.data.Dataset):
    def __init__(self, subfolder, classes=CIFAR_CLASSES):
        self.classes = classes

        self.images = []

        base_path = "data/cifar10"
        for c in classes:
            self.images += [(pth, c)
                            for pth in os.listdir(os.path.join(base_path, subfolder, c))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(os.path.join("data", self.images[idx][0]))
        return F.pil_to_tensor(im), self.images[idx][1]


def get_pytorch_dataloaders():

    transform = transforms.Compose(
        [transforms.ToTensor()])

    batch_size = 1

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader


def save_dataset(trainloader, testloader):
    classes = ['plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    pths = ["data/cifar10/cifar_10_segmented_train", "data/cifar10/cifar_10_segmented_test"]

    os.mkdir("data/cifar10")
    for pth in pths:
        os.mkdir(pth)

    counter_set = {c: 0 for c in classes}
    for loader, save_pth in zip([trainloader, testloader], pths):
        for x, y in loader:
            cls = classes[int(y.cpu().detach().numpy())]
            pth = os.path.join(save_pth, cls)
            counter_set[cls] += 1
            if not os.path.exists(pth):
                os.mkdir(pth)
            image = F.to_pil_image(torch.squeeze(x))
            image.save(os.path.join(pth, str(counter_set[cls]) + ".png"))


def create_dataloaders(config):
    if not os.path.exists(os.path.join("data", "cifar10")):
        trainloader, testloader = get_pytorch_dataloaders()
        save_dataset(trainloader, testloader)
    # insert logic for creating the dataloaders
    train = torch.utils.data.DataLoader(
        CIFARDataset("cifar_10_segmented_train",
                        classes=config["classes"]),
        batch_size=config["batch_size"], shuffle=True)

    test = torch.utils.data.DataLoader(
        CIFARDataset("cifar_10_segmented_test",
                        classes=config["classes"]),
        batch_size=config["batch_size"], shuffle=True)
    return train, test


if __name__ == "__main__":
    create_dataloaders({"batch_size":32, "classes":CIFAR_CLASSES})
    print([sum([int(i.split(".")[0]) for i in os.listdir("data/cifar10/cifar_10_segmented_train/" + cls)]) for cls in CIFAR_CLASSES])
    print([sum([int(i.split(".")[0]) for i in os.listdir("data/cifar10/cifar_10_segmented_test/" + cls)]) for cls in CIFAR_CLASSES])
