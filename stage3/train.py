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

EPOCHS = 20
PYRAMID_HEIGHT = 6

if IS_TOPS:
    stage = 'tops'
    in_channels = 33
else:
    stage = 'bottoms'
    in_channels = 9
dataroot = '/home/fashionteam/dataset_MVC_'+stage
dataroot_mask = osp.join(PWD,"result/warped_mask",stage)
dataroot_cloth = osp.join(PWD,"result/warped_cloth",stage)
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = '/home/fashionteam/ClothFlow/stage3/checkpoints/'+stage
exp = 'train/'+stage

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", type=str, default=dataroot_mask)
    parser.add_argument("--dataroot_cloth", type=str, default=dataroot_cloth)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = stage)
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/fashionteam/ClothFlow/stage3/checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
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
    model.cuda()
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt, is_tops=IS_TOPS)
    train_loader = CFDataLoader(opt, train_dataset)

    writer = SummaryWriter()
    rLoss = renderLoss()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            warped = inputs['warped'].cuda()
            answer = inputs['image'].cuda()
            off_cloth = inputs['off_cloth'].cuda()
            pants = inputs['crop_pants'].cuda()
            cloth_mask = inputs['cloth_mask'].cuda()
            if IS_TOPS:
                head = inputs['head'].cuda()
                pose = inputs['pose'].cuda()
            name = inputs['name']
            if IS_TOPS:
                result = model(con_cloth,off_cloth,pants,warped,cloth_mask,head,pose)
            else:
                result = model(con_cloth,off_cloth,warped)

            if cnt % 50 == 0:
                WriteImage(writer,"GT", answer, cnt)
                #WriteImage(writer,"shape", shape, cnt)
                WriteImage(writer,"warped", warped, cnt)
                WriteImage(writer,"con_cloth", con_cloth, cnt)
                WriteImage(writer,"off_cloth", off_cloth, cnt)
                if IS_TOPS:
                    WriteImage(writer,"Head", head, cnt)
                else:
                    WriteImage(writer,"pants", pants, cnt)
                    WriteImage(writer,"cloth_mask", cloth_mask, cnt)
                WriteImage(writer,"Result", result, cnt)
            
            if IS_TOPS and epoch == 0:
                optimizer.zero_grad()
                loss, percept, style = rLoss(result,head)
                loss.backward()
                optimizer.step()
            else:	
                optimizer.zero_grad()
                loss, percept, style = rLoss(result,answer,cloth_mask)
                loss.backward()
                optimizer.step()

            # if (step+1) % opt.display_count == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
                #     100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))

            writer.add_scalar("loss/loss", loss, cnt)
            writer.add_scalar("loss/percept", percept, cnt)
            writer.add_scalar("loss/style", style, cnt)

            writer.close()

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.exp, '%d_%05d.pth' % (epoch, (step+1))))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= '0,1'

    opt = get_opt()
    train(opt)
