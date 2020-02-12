from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as Ftnl
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from Models.UNetS3 import *
from Models.LossS3 import *
from dataloader_MVC import *
import argparse
from tqdm import tqdm
from torchvision.utils import save_image

from tensorboardX import SummaryWriter
import sys
import time

EPOCHS = 15
PYRAMID_HEIGHT = 5
IS_TOPS = False # TOPS or BOTTOMS 

if IS_TOPS:
    dataroot = '/home/fashionteam/dataset_MVC_tops/'
    datalist = 'MVCtops_pair.txt'
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage1/checkpoints/tops/'
    exp = 'train/tops/'
    in_channels = 22
else:
    dataroot = '/home/fashionteam/dataset_MVC_bottoms/'
    datalist = 'MVCbottoms_pair.txt'
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage1/checkpoints/bottoms/'
    exp = 'train/bottoms/'
    in_channels = 2
    
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=10)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "stage1")
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--exp", type=str, default = exp)
    

    opt = parser.parse_args()
    return opt

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
       print("ERROR")
       return 
    model.load_state_dict(torch.load(checkpoint_path))	
    model.cuda()

def train(opt):

    model = UNet(opt, depth=PYRAMID_HEIGHT, in_channels=in_channels)
    model = nn.DataParallel(model)
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt, is_tops=IS_TOPS)
    train_loader = CFDataLoader(opt, train_dataset)

    os.rmdir("/home/fashionteam/ClothFlow/stage1/runs/"+opt.exp)
    writer = SummaryWriter("/home/fashionteam/ClothFlow/stage1/runs/"+opt.exp)
    rLoss = renderLoss()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs["cloth_mask"].cuda()
            name = inputs['name']
            if IS_TOPS:
                tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
                pose = inputs['pose'].cuda()
            else:
                tar_cloth_mask = inputs['crop_pants_mask'].cuda()
                tar_body_mask = inputs['target_body_shape'].cuda()
            
            if IS_TOPS:
                result = model(con_cloth, con_cloth_mask, pose)
            else:
                result = model(con_cloth_mask, tar_body_mask, None)
            
            optimizer.zero_grad()
            loss = rLoss(result, tar_cloth_mask)
            loss.backward()
            optimizer.step()

            if (step+1) % opt.display_count == 0:
                writer.add_images("cloth", con_cloth, cnt)
                writer.add_images("mask", con_cloth_mask, cnt)
                writer.add_images("GT", tar_cloth_mask, cnt)
                writer.add_images("Result", result, cnt)
                writer.add_scalar("loss/loss", loss, cnt)
                writer.close()

            # if (step+1) % opt.display_count == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
            #         100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))


            if (step+1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.exp, '%d_%05d.pth' % (epoch, step+1)))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

    opt = get_opt()
    train(opt)
