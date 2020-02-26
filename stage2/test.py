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
from Models.networks import *
from Models.ClothNormalize_proj import *
from dataloader_MVC import *

PYRAMID_HEIGHT = 5

if IS_TOPS:
    stage = 'tops'
    nc = 2
    checkpoint = 'backup/stage2_top_512.pth'
    init_CN = 'backup/CN_top.pth'
else:
    stage = 'bottoms'
    nc = 2
    checkpoint = 'backup/stage2_bot_512.pth'
    init_CN = 'backup/CN_bot.pth'

dataroot = '/home/fashionteam/dataset_MVC_'+stage
dataroot_mask = '/home/fashionteam/ClothFlow/result/warped_mask/'+stage
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/'+stage
result_dir = '/home/fashionteam/ClothFlow/result/warped_cloth_CN/'+stage
exp = 'train/'+stage

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=2)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", default = dataroot_mask)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = stage)
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default=result_dir, help='save result infos')
    parser.add_argument('--checkpoint', type=str, default=checkpoint_dir, help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--save_count", type=int, default = 1)
    parser.add_argument("--save_img_count", type=int, default = 1)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    parser.add_argument("--smt_loss", type=float, default=2)
    parser.add_argument("--perc_loss", type=float, default=1)
    parser.add_argument("--struct_loss", type=float, default=10)
    parser.add_argument("--stat_loss", type=float, default=-1)
    parser.add_argument("--abs_loss", type=float, default=0)
    parser.add_argument("--save_dir", type=str, default="npz")
    parser.add_argument("--naming", type=str, default="default")

    opt = parser.parse_args()
    return opt

def test(opt):
    model = FlowNet(PYRAMID_HEIGHT)
    model = nn.DataParallel(model)

    # load_checkpoint(model, opt.checkpoint)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    test_dataset = CFDataset(opt, is_tops=IS_TOPS)
    test_loader = CFDataLoader(opt, test_dataset)

    model.cuda()
    model.eval()

    Flow = FlowLoss(opt).cuda()

    if not osp.isdir(opt.result_dir):
        os.makedirs(opt.result_dir)

    theta_generator = ClothNormalizer(depth=PYRAMID_HEIGHT, nc=nc)
    load_checkpoint(theta_generator,init_CN)
    theta_generator.cuda()
    theta_generator.eval()

    writer = SummaryWriter()
    
    rangelist = range(len(test_loader.dataset)//opt.batch_size + 1)
    for step in tqdm(rangelist, desc='step') if TENSORBOARD else rangelist:
        cnt = step + 1
        inputs = test_loader.next_batch()

        name = inputs['name']
        con_cloth = inputs['cloth'].cuda()
        con_cloth_mask = inputs['cloth_mask'].cuda()
        tar_cloth = inputs['crop_cloth'].cuda()
        tar_cloth_mask = inputs['crop_cloth_mask'].cuda()

        theta = theta_generator(con_cloth_mask, tar_cloth_mask)
        
        grid1 = projection_grid(theta, con_cloth_mask.shape)
        grid2 = projection_grid(theta, con_cloth.shape)
        con_cloth_mask = Ft.grid_sample(con_cloth_mask , grid1).detach()
        con_cloth = Ft.grid_sample(con_cloth , grid2,padding_mode="border").detach()
    
        [F, warp_cloth, warp_mask] = model(torch.cat([con_cloth, con_cloth_mask], 1), tar_cloth_mask)
        
        # if cnt % opt.save_img_count == 0:
        #     writer.add_images("con_cloth", con_cloth, cnt)
        #     writer.add_images("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
        #     writer.add_images("tar_cloth", tar_cloth, cnt)
        #     writer.add_images("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")
        #     writer.add_images("warp_cloth", warp_cloth, cnt)
        #     writer.add_images("warp_mask", warp_mask, cnt, dataformats="NCHW")
                
        loss, roi_perc, struct, smt, stat, abs = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth, con_cloth_mask)

        if cnt % opt.display_count == 0:
            print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 cnt, len(test_loader.dataset),
                100. * cnt / len(test_loader.dataset), 0))

        save_images(con_cloth, name, opt.result_dir)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    opt = get_opt()
    test(opt)