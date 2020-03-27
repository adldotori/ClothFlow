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

PYRAMID_HEIGHT = 5

if IS_TOPS:
    stage = 'tops'
    in_channels = 33
    checkpoint = 'backup/stage3_top_512.pth'
else:
    stage = 'bottoms'
    in_channels = 9
    checkpoint = 'backup/stage3_bot_512.pth'

dataroot = '/home/fashionteam/dataset_MVC_'+stage
dataroot_mask = osp.join(PWD,"result/warped_mask",stage)
dataroot_cloth = osp.join(PWD,"result/warped_cloth",stage)
datalist = 'train_MVC'+stage+'_pair.txt'
result_dir = osp.join(PWD,'result/final',stage)
exp = 'train/'+stage

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    
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
    parser.add_argument('--result_dir', type=str, default=result_dir, help='save result infos')
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
    parser.add_argument("--save_dir", type=str, default="npz")
    

    opt = parser.parse_args()
    return opt

def test(opt):

    model = UNet(opt, 24, 5)
    model = nn.DataParallel(model)
    load_checkpoint(model, opt.checkpoint)

    model.cuda()
    model.train()

    test_dataset = CFDataset(opt)
    test_loader = CFDataLoader(opt, test_dataset)

    writer = SummaryWriter()
    rLoss = renderLoss()

    if not osp.isdir(opt.result_dir):
        os.makedirs(opt.result_dir)

    rangelist = range(len(test_loader.dataset)//opt.batch_size + 1)
    for step in tqdm(rangelist, desc='step') if TENSORBOARD else rangelist:
        cnt = step + 1
        
        inputs = test_loader.next_batch()
        
        con_cloth = inputs['cloth'].cuda()
        warped = inputs['warped'].cuda()
        answer = inputs['image'].cuda()
        off_cloth = inputs['off_cloth'].cuda()
        pose = inputs['pose'].cuda()
        head = inputs['head'].cuda()
        pants = inputs['crop_pants'].cuda()
        name = inputs['name']

        result = model(con_cloth,off_cloth,pose,warped,head,pants)

        WriteImage(writer,"GT", answer, cnt)
        WriteImage(writer,"warped", warped, cnt)
        WriteImage(writer,"con_cloth", con_cloth, cnt)#, dataformats="NCHW")
        WriteImage(writer,"off_cloth", off_cloth, cnt)
        WriteImage(writer,"Result", result, cnt)

        loss, percept, style = rLoss(result,answer)

        if not TENSORBOARD:
            if cnt % opt.display_count == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, (step+1) * 1, len(test_loader.dataset)//opt.batch_size + 1,
                    100. * (step+1) / (len(test_loader.dataset)//opt.batch_size + 1), loss.item()))

        writer.add_scalar("loss/loss", loss, cnt)
        writer.add_scalar("loss/percept", percept, cnt)
        writer.add_scalar("loss/style", style, cnt)

        writer.close()

        save_images(result,name,result_dir)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

    opt = get_opt()
    test(opt)
