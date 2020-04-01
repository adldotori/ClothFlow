from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as Ft
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp
import sys
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import time

sys.path.append('..')
from utils import *
from Models.UNetS3 import *
from Models.LossS3 import *
from dataloader_MVC import *

EPOCHS = 200
PYRAMID_HEIGHT = 5
IS_TOPS = True

if IS_TOPS:
    stage = 'tops'
    in_channels = 3
    checkpoint = None
    # checkpoint = 'fullface/checkpoints/ck11.pth'
else:
    stage = 'bottomqs'
    in_channels = 9
dataroot = '/home/fashionteam/dataset_MVC_tops/'
dataroot_mask = osp.join(PWD,"result_viton/warped_mask",stage)
dataroot_cloth = osp.join(PWD,"result_viton/warped_cloth",stage)
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = '/home/fashionteam/ClothFlow/fullface/checkpoints/'

exp = 'train/'+stage

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", type=str, default=dataroot_mask)
    parser.add_argument("--dataroot_cloth", type=str, default=dataroot_cloth)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = stage)
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 10)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 1000)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    parser.add_argument("--naming", type=str, default="default")
    parser.add_argument("--save_dir", type=str, default="npz")
    parser.add_argument("--exp", type=str, default=exp)
    

    opt = parser.parse_args()
    return opt

def get_test_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", type=str, default=dataroot_mask)
    parser.add_argument("--dataroot_cloth", type=str, default=dataroot_cloth)
    parser.add_argument("--datamode", default = "test")
    parser.add_argument("--stage", default = stage)
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 10)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 1000)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    parser.add_argument("--naming", type=str, default="default")
    parser.add_argument("--save_dir", type=str, default="npz")
    parser.add_argument("--exp", type=str, default=exp)
    

    opt = parser.parse_args()
    return opt

def train(opt):
    model = UNet(opt, in_channels, PYRAMID_HEIGHT)
    model = nn.DataParallel(model)
    load_checkpoint(model, opt.checkpoint)
    model.cuda()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)
    test_dataset = CFDataset(get_test_opt())
    test_loader = CFDataLoader(get_test_opt(), test_dataset)

    writer = SummaryWriter()
    rLoss = renderLoss()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            lack = inputs['lack'].cuda()
            full = inputs['full'].cuda()
            face = inputs['face'].cuda()

            result = model(lack)
            WriteImage(writer,"lack", lack, cnt)
            WriteImage(writer,"full", full, cnt)
            WriteImage(writer,"face", face, cnt)
            WriteImage(writer,"result", result, cnt)

            optimizer.zero_grad()
            loss, percept, style = rLoss(result, full)
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/loss", loss, cnt)
            writer.add_scalar("loss/percept", percept, cnt)
            writer.add_scalar("loss/style", style, cnt)

            writer.close()

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, 'checkpoint_2_%d.pth' % (cnt)))
        
        inputs = test_loader.next_batch()
        lack = inputs['lack'].cuda()
        full = inputs['full'].cuda()
        face = inputs['face'].cuda()

        result = model(lack)

        loss, percept, style = rLoss(result, full)
        print("Loss : %.2f\n", loss)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '1,2,3'

    opt = get_opt()
    train(opt)