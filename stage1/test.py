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
from dataloader_MVC import *

PYRAMID_HEIGHT = 5
IS_TOPS = True

if IS_TOPS:
    stage = 'tops'
    in_channels = 22
    checkpoint = 'backup/stage1_top_512.pth'
else:
    stage = 'bottoms'
    in_channels = 2
    checkpoint = 'backup/stage1_bot_512.pth'
if REAL_TEST:
    dataroot = '/home/fashionteam/dataset_MVC_'+stage
    datalist = 'test_MVC'+stage+'_pair.txt'
    result_dir = osp.join(PWD,'test/warped_mask/',stage)
else:
    dataroot = '/home/fashionteam/dataset_MVC_'+stage
    datalist = 'train_MVC'+stage+'_pair.txt'
    result_dir = osp.join(PWD,'result/warped_mask_6/',stage)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "stage1")
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
    

    opt = parser.parse_args()
    return opt

def test(opt):
    model = UNet(opt, depth=PYRAMID_HEIGHT, in_channels=in_channels)
    model = nn.DataParallel(model)
    load_checkpoint(model, opt.checkpoint)
    model.cuda()
    model.eval()
    test_dataset = CFDataset(opt, is_tops=IS_TOPS)
    test_loader = CFDataLoader(opt, test_dataset)

    writer = SummaryWriter()
    rLoss = renderLoss()

    if not osp.isdir(opt.result_dir):
        os.makedirs(opt.result_dir)

    for step in tqdm(range(len(test_loader.dataset)//opt.batch_size + 1), desc='step'):

        cnt = step + 1
        
        inputs = test_loader.next_batch()
        
        con_cloth = inputs['cloth'].cuda()
        con_cloth_mask = inputs['cloth_mask'].cuda()
        name = inputs['name']
        if IS_TOPS:
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
            pose = inputs['pose'].cuda()
        else:
            tar_cloth_mask = inputs['crop_pants_mask'].cuda() #answer   
            tar_body_mask = inputs['target_body_shape'].cuda()

        if IS_TOPS:
            print(con_cloth.shape, con_cloth_mask.shape, pose.shape)
            result = model(con_cloth, con_cloth_mask, pose, IS_TOPS)
        else:
            result = model(con_cloth_mask, tar_body_mask, None, IS_TOPS)

        loss = rLoss(result, tar_cloth_mask)

        if (step+1) % opt.display_count == 0:
            writer.add_images("GT", tar_cloth_mask, cnt)
            writer.add_images("cloth", con_cloth, cnt)
            writer.add_images("mask", con_cloth_mask, cnt)
            writer.add_images("Result", result, cnt)
            writer.add_scalar("loss/loss", loss, cnt)
            writer.close()

        # if (step+1) % opt.display_count == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch+1, (step+1) * 1, len(test_loader.dataset)//opt.batch_size + 1,
        #         100. * (step+1) / (len(test_loader.dataset)//opt.batch_size + 1), loss.item()))
        # result = (result > -0.9).type(torch.float32)
        save_images(result, name, opt.result_dir)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,2,3"

    opt = get_opt()
    test(opt)
