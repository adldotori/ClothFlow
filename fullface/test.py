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
from dataloader_test import *

PYRAMID_HEIGHT = 5

stage = 'tops'
in_channels = 3
checkpoint = 'fullface/checkpoints/ck11.pth'

dataroot = '/home/fashionteam/final_test/'
result_dir = '/home/fashionteam/final_test/'
exp = 'train/'+stage
        
# def save_images(img_tensors, img_names, save_dir):
#     for img_tensor, img_name in zip(img_tensors, img_names):
#         img = img_tensor.clone()* 255
#         if img.shape[0] == 1:
#             img = img[0,:,:]
#         else:
#             img = img.transpose(0, 1).transpose(1, 2)
#         img = img.cpu().clamp(0,255)

#         # array = tensor.numpy().astype('uint8')
#         img = img.detach().numpy().astype('uint8')
#         image = Image.fromarray(img)
#         # image.show()
#         image.save(os.path.join(save_dir, img_name + '.jpg'))

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = stage)
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

    model = UNet(opt, in_channels, PYRAMID_HEIGHT)
    model = nn.DataParallel(model)
    load_checkpoint(model, opt.checkpoint)

    model.cuda()
    model.eval()

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
        
        lack = inputs['lack'].cuda()
        name = inputs['name']
        

        result = model(lack)

        WriteImage(writer, "lack", lack, cnt, 1)
        WriteImage(writer, "result", result, cnt, 1)
        writer.close()

        save_images(result,['image_'],result_dir+name[0])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0"

    opt = get_opt()
    test(opt)

