from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models.networks import *
from dataloader import *

INPUT_SIZE = (1024, 1024)
EPOCHS = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(epoch):
    model = FlowNet(4, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train_loader = ClothDataLoader(opt, train_dataset)

    model.cuda()
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        model(data)
        model.backward()
        optimizer.step()
        result = model.current_results()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), result['loss']))

if __name__ == '__main__':
    for i in range(EPOCHS):
        train(i)