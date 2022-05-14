import torch
import torch.nn as nn
import numpy as np

import os
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

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
    
    pths = ["data/cifar_10_segmented_train", "data/cifar_10_segmented_test"]

    counter_set = {c:0 for c in classes}
    for loader, save_pth in zip([trainloader, testloader], pths):
        for x, y in loader:
            cls = classes[int(y.cpu().detach().numpy())]
            pth = os.path.join(save_pth, cls)
            counter_set[cls] += 1
            if not os.path.exists(pth):
                os.mkdir(pth)  
            image = F.to_pil_image(torch.squeeze(x))
            image.save(os.path.join(pth, str(counter_set[cls]) +".png"))
        
def create_dataloaders_cifar(config):
    if not os.path.exists("data", "cifar10"):
        trainloader, testloader = get_pytorch_dataloaders()
        save_dataset(trainloader, testloader)
    else:
        # insert logic for creating the dataloaders
        raise NotImplementedError   
