import torch
import torch.nn as nn
import numpy as np

import os
from PIL import Image

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class AnimeFaceDataset(torch.utils.data.Dataset):
    def __init__(self):

        self.images = []
        base_path = "data/animefacedataset/images"
        self.images += [(os.path.join(base_path, pth), 0)
                        for pth in os.listdir(os.path.join(base_path))]
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((88, 88))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = Image.open(self.images[idx][0])
        return self.transforms((im)), self.images[idx][1]


def create_dataloaders(batch_size):

    # insert logic for creating the dataloaders
    train = torch.utils.data.DataLoader(
        AnimeFaceDataset(),
        batch_size=batch_size, shuffle=True)

    test = torch.utils.data.DataLoader(
        AnimeFaceDataset(),
        batch_size=batch_size, shuffle=True)
    return train, test


if __name__ == "__main__":
    a, b = create_dataloaders(**{"batch_size": 1})
    min_a = (np.inf, np.inf)

    for i in a:
        assert ((i[0].shape[2] == 88) and (i[0].shape[3] == 88))
    for i in b:
        assert ((i[0].shape[2] == 88) and (i[0].shape[3] == 88))