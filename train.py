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
from dataloader_viton import *
import argparse

INPUT_SIZE = (192, 256)
EPOCHS = 2000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/Taeho/VITON/train")
    parser.add_argument("--datamode", default = "")
    parser.add_argument("--stage", default = "1")
    parser.add_argument("--data_list", default = "MVCup_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--keep_step", type=int, default = 100000)
    parser.add_argument("--decay_step", type=int, default = 100000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def train(opt, epoch):
    model = FlowNet(4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)

    model.cuda()
    model.train()

    for step in range(opt.keep_step):
        inputs = train_loader.next_batch()

        con_cloth = inputs['cloth'].to(device)
        con_cloth_mask = inputs['cloth_mask'].to(device)
        tar_cloth = inputs['crop_cloth'].to(device)
        tar_cloth_mask = inputs['crop_cloth_mask'].to(device)

        optimizer.zero_grad()
        print(torch.cat([con_cloth, con_cloth_mask], 1).shape, tar_cloth_mask.shape)
        result = model(torch.cat([con_cloth, con_cloth_mask], 1), tar_cloth_mask)
        # TODO : save result
        optimizer.zero_grad()
        model.backward(tar_cloth)
        optimizer.step()
        result = model.current_results()

        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), result['loss']))

if __name__ == '__main__':
    opt = get_opt()
    for i in range(EPOCHS):
        train(opt, i)