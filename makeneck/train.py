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
sys.path.append('/home/fashionteam/ClothFlow/stage2')
from utils import *
from Models.ClothNormalize_proj import *
from Models.UNetS3 import *
from Models.LossS3 import *
from dataloader_viton import *

EPOCHS = 200
PYRAMID_HEIGHT = 5
IS_TOPS = True

if IS_TOPS:
    stage = 'tops'
    in_channels = 22
    checkpoint = None
    checkpoint = 'makeneck/checkpoints/checkpoint_3_0.pth'
else:
    stage = 'bottomqs'
    in_channels = 9
dataroot = '/home/fashionteam/viton_512'
init_CN = 'stage2/checkpoints/CN/train/tops/Epoch:14_00466.pth'
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = '/home/fashionteam/ClothFlow/makeneck/checkpoints/'

exp = 'train/'+stage

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = stage)
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.00002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    parser.add_argument("--smt_loss", type=float, default=2)
    parser.add_argument("--perc_loss", type=float, default=1)
    parser.add_argument("--struct_loss", type=float, default=1.7*10)
    parser.add_argument("--stat_loss", type=float, default=0)
    parser.add_argument("--abs_loss", type=float, default=0)
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

    writer = SummaryWriter()
    l1Loss = nn.L1Loss()
    rLoss = renderLoss()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            nonneck = inputs['nonneck'].cuda()
            answer = inputs['image'].cuda()
            parse = inputs['parse'].cuda()
            pose = inputs['pose'].cuda()
            head = inputs['head'].cuda()

            result = model(pose, parse, nonneck)
            WriteImage(writer,"GT", answer, cnt)
            WriteImage(writer,"nonneck", nonneck, cnt)
            WriteImage(writer,"parse", parse, cnt)
            WriteImage(writer,"result", result, cnt)
            WriteImage(writer,"head", head, cnt)

            optimizer.zero_grad()
            l1_ = l1Loss(result, answer)
            loss_, percept, style = rLoss(result, answer)
            loss = loss_# * 0.001 + l1_
            loss.backward()
            optimizer.step()

            writer.add_scalar("loss/loss", loss, cnt)
            # writer.add_scalar("loss/percept", percept, cnt)
            # writer.add_scalar("loss/style", style, cnt)

            writer.close()

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, 'checkpoint_4_%d.pth' % (cnt%3)))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '1,2,3'

    opt = get_opt()
    train(opt)
