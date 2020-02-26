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
from Models.networks_neck import *
from Models.LossS3 import *
from Models.net_canny import *
from Models.loss_canny import *
from dataloader_neck import *


EPOCHS = 15
PYRAMID_HEIGHT = 5
NUM_STAGE = str(1)

if IS_TOPS:
    stage = 'tops'
    in_channels = 23
else:
    stage = 'bottoms'
    in_channels = 2
dataroot = '/home/fashionteam/dataset_MVC_'+stage
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = osp.join(PWD,'stage'+NUM_STAGE, "neck", 'checkpoints',stage)
runs = osp.join(PWD,'stage'+NUM_STAGE,"neck", 'runs')
exp = osp.join('train',stage)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument("--stage", default = "train")
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument("--result_dir", default = "result")
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--exp", type=str, default = exp)
    

    opt = parser.parse_args()
    return opt

def train(opt):
    model = UNet(opt, depth=PYRAMID_HEIGHT, in_channels=in_channels)
    model = nn.DataParallel(model)
    model.cuda()
    model.train()

    canny = Canny()
    canny.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt, is_tops=IS_TOPS)
    train_loader = CFDataLoader(opt, train_dataset)

    if os.path.isdir(runs):
        os.system('rm -r '+runs+opt.exp)
    if not (os.path.exists(runs+opt.exp)): os.makedirs(runs+opt.exp)
    writer = SummaryWriter(runs+opt.exp)

    rLoss = renderLoss()
    cLoss = cannyLoss()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        if (epoch <= 5): style = 0.0
        else: style = 1/300
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs["cloth_mask"].cuda()
            name = inputs['name']
            if IS_TOPS:
                tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
                pose = inputs['pose'].cuda()
                shape = inputs['shape'].cuda()
            else:
                tar_cloth_mask = inputs['crop_pants_mask'].cuda()
                tar_body_mask = inputs['target_body_shape'].cuda()
            
            if IS_TOPS:
                result = model(con_cloth, con_cloth_mask, pose, IS_TOPS, shape)
            else:
                result = model(con_cloth_mask, tar_body_mask, None, IS_TOPS)
            
            optimizer.zero_grad()
            
            img1 = canny(result)
            img2 = canny(tar_cloth_mask)
				
            writer.add_images("result_canny", img1, cnt)
            writer.add_images("GT_canny", img2, cnt)
            if IS_TOPS:
                shape = shape.view(shape.shape[0], 1, shape.shape[1], shape.shape[2])
                writer.add_images("shape", shape, cnt)

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
            c_loss, _ = cLoss(img1, img2)
            loss += c_loss * style
			
            loss.backward()
            optimizer.step()

            if (step+1) % opt.display_count == 0:
                writer.add_images("cloth", con_cloth, cnt)
                writer.add_images("mask", con_cloth_mask, cnt)
                writer.add_images("GT", tar_cloth_mask, cnt)
                if not IS_TOPS:
                    writer.add_images("tar_body", tar_body_mask, cnt)
                writer.add_images("Result", result, cnt)
                writer.add_scalar("loss/loss", loss, cnt)
                # writer.add_scalar("loss/c_loss/style", style, cnt)
                # writer.add_scalar("loss/c_loss/mse", mse, cnt)
                writer.add_scalar("loss/c_loss", c_loss*style, cnt)
                writer.add_scalar("loss/r_loss", r_loss, cnt)
                writer.close()

            # if (step+1) % opt.display_count == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
            #         100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))


            if (step+1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.exp, '%d_%05d.pth' % (epoch, step+1)))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"

    opt = get_opt()
    train(opt)
