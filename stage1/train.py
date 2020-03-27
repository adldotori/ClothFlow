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
from Models.net_canny import *
from Models.loss_canny import *
from dataloader_viton import *


EPOCHS = 200
PYRAMID_HEIGHT = 5
NUM_STAGE = str(1)
IS_TOPS = True

if IS_TOPS:
    stage = 'tops'
    in_channels = 22
else:
    stage = 'bottoms'
    in_channels = 2
dataroot = '/home/fashionteam/viton_512/'
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = osp.join(PWD,'stage'+NUM_STAGE,'checkpoints',stage)
runs = osp.join(PWD,'stage'+NUM_STAGE,'runs','train',stage)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=2)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--runs", type=str, default=runs)
    

    opt = parser.parse_args()
    return opt

def train(opt):
    model = UNet(opt, depth=PYRAMID_HEIGHT, in_channels=in_channels)
    model = nn.DataParallel(model)
    # load_checkpoint(model, 'stage1/checkpoints/tops/checkpoint_7_86000.pth')
    model.cuda()
    model.train()

    canny = Canny()
    canny.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)

    if os.path.isdir(opt.runs):
        os.system('rm -r '+opt.runs)
    os.makedirs(opt.runs)
    writer = SummaryWriter(opt.runs)

    rLoss = renderLoss()
    cLoss = WeightedMSE()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs["cloth_mask"].cuda()
            name = inputs['name']
            tar_cloth = inputs['crop_cloth'].cuda()
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
            # tar_body_mask = inputs['tar_body_mask'].cuda()
            pose = inputs['pose'].cuda()
            arms_mask = inputs['arms_mask'].cuda()

            result = model(pose, con_cloth, con_cloth_mask,IS_TOPS)
            # result = model(con_cloth, con_cloth_mask, pose, IS_TOPS)

            optimizer.zero_grad()
            
            img1 = canny(result)
            img2 = canny(tar_cloth)
				
            writer.add_images("result_canny", img1, cnt)
            writer.add_images("GT_canny", img2, cnt)

            # style, mse = cLoss(img1, img2)
            # if epoch <= 4:
            #     style = style * 0.0
            #     mse = mse * 0.0
            # else:
            #     style = style / 1000	
            #     mse = mse * 100
            r_loss = rLoss(result, tar_cloth_mask)

            # c_loss = style + mse
            # loss = c_loss + r_loss

            loss = r_loss

            loss.backward()
            optimizer.step()

            if (step+1) % opt.display_count == 0:
                writer.add_images("cloth", con_cloth, cnt)
                writer.add_images("mask", con_cloth_mask, cnt)
                writer.add_images("GT", tar_cloth_mask, cnt)
                writer.add_images('tar_cloth', tar_cloth, cnt)
                # writer.add_images("tar_body", tar_body_mask, cnt)
                writer.add_images("Result", result, cnt)
                writer.add_scalar("loss/loss", loss, cnt)
                writer.add_scalar("loss/r_loss", r_loss, cnt)
                writer.close()

            # if (step+1) % opt.display_count == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
            #         100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))


            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, 'checkpoint_10_%d.pth' % cnt))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"

    opt = get_opt()
    train(opt)
