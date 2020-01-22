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
from tqdm import tqdm_notebook

from tensorboardX import SummaryWriter

INPUT_SIZE = (192, 256)
<<<<<<< HEAD
EPOCHS = 15
PYRAMID_HEIGHT = 4

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
=======
EPOCHS = 10
PYRAMID_HEIGHT = 5

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
>>>>>>> 832da05656ead46309225214d33574039111be94

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_resize/train/")
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
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.to(device)

def train(opt):
    model = FlowNet(PYRAMID_HEIGHT, 4, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)

    model.to(device)
    model.train()
    Flow = FlowLoss().to(device)

    writer = SummaryWriter()

    for epoch in tqdm_notebook(range(EPOCHS), desc='EPOCH'):
        for step in tqdm_notebook(range(len(train_loader.dataset)), desc='step'):
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].to(device)
            con_cloth_mask = inputs['cloth_mask'].to(device)
            tar_cloth = inputs['crop_cloth'].to(device) 
            tar_cloth_mask = inputs['crop_cloth_mask'].to(device)

            writer.add_image("con_cloth", con_cloth, step, dataformats="NCHW")
            writer.add_image("con_cloth_mask", con_cloth_mask, step, dataformats="NCHW")
            writer.add_image("tar_cloth", tar_cloth, step, dataformats="NCHW")
            writer.add_image("tar_cloth_mask", tar_cloth_mask, step, dataformats="NCHW")

            [F, warp_cloth, warp_mask] = model(torch.cat([con_cloth, con_cloth_mask], 1), tar_cloth_mask)
            optimizer.zero_grad()
            loss, roi_perc, struct, smt = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth)
            loss.backward()
            optimizer.step()

            # if (step+1) % opt.display_count == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch+1, (step+1) * 1, len(train_loader.dataset),
                #     100. * (step+1) / len(train_loader.dataset), loss.item()))
            writer.add_scalar("loss/roi_perc", roi_perc, step)
            writer.add_scalar("loss/struct", struct, step)
            writer.add_scalar("loss/smt", smt, step)
            writer.close()

            # if (step+1) % opt.display_count == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, (step+1) * 1, len(train_loader.dataset),
            #         100. * (step+1) / len(train_loader.dataset), loss.item()))

            if (epoch * len(train_loader.dataset) + step + 1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.stage, '%d_%05d.pth' % (epoch, (step+1))))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"

    opt = get_opt()
    train(opt)
