from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as Ft
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

from tensorboardX import SummaryWriter

from Models.networks import *
from dataloader_viton import *
from Models.ClothNormalize_proj import *
sys.path.append('..')
from utils import *

PYRAMID_HEIGHT = 5
DATASET = 'MVC'
IS_TOPS = False

if DATASET is 'MVC':
    from dataloader_MVC import *
    if IS_TOPS:
        stage = 'tops'
        nc = 36
    else:
        stage = 'bottoms'
        nc = 2
    dataroot = '/home/fashionteam/dataset_MVC_'+stage
    dataroot_mask = '/home/fashionteam/ClothFlow/result/warped_mask/'+stage
    datalist = 'MVC'+stage+'_pair.txt'
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/'+stage
    result_dir = '/home/fashionteam/ClothFlow/result/warped_cloth/'+stage
    exp = 'train/'+stage
else:
    from dataloader_viton import *
    dataroot = '/home/fashionteam/viton_resize/'
    datalist = ''
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/tops/'
    result_dir = '/home/fashionteam/ClothFlow/result/warped_cloth/viton/'
    exp = 'train/tops/'

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
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
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 1)
    parser.add_argument("--save_img_count", type=int, default = 50)
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

def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        # array = tensor.numpy().astype('uint8')
        array = tensor.detach().numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
        image = Image.fromarray(array)
        # image.show()
        image.save(os.path.join(save_dir, img_name + '.jpg'))

def test(opt):
    model = FlowNet(PYRAMID_HEIGHT)
    model = nn.DataParallel(model)

    load_checkpoint(model, "stage2/checkpoints/default/tops/1_02700.pth")
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    test_dataset = CFDataset(opt, is_tops=IS_TOPS)
    test_loader = CFDataLoader(opt, test_dataset)

    model.cuda()
    model.eval()

    Flow = FlowLoss(opt).cuda()

    if not osp.isdir(opt.result_dir):
        os.makedirs(opt.result_dir)

    theta_generator = ClothNormalizer(depth=PYRAMID_HEIGHT, nc=nc)
    load_checkpoint(theta_generator,"stage2/checkpoints/ClothNormalizer_proj_bot/Epoch:3_00210.pth")
    theta_generator.cuda()
    theta_generator.eval()

    writer = SummaryWriter(comment = "_" + opt.naming)

    for step in range(len(test_loader.dataset)):
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
        
        if (step+1) % opt.save_img_count == 0:
            writer.add_images("con_cloth", con_cloth, cnt)
            writer.add_images("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
            writer.add_images("tar_cloth", tar_cloth, cnt)
            writer.add_images("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")
            writer.add_images("warp_cloth", warp_cloth, cnt)
            writer.add_images("warp_mask", warp_mask, cnt, dataformats="NCHW")
                
        loss, roi_perc, struct, smt, stat, abs = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth, con_cloth_mask)

        if (step+1) % opt.display_count == 0:
            print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 (step+1) * 1, len(test_loader.dataset),
                100. * (step+1) / len(test_loader.dataset), loss.item()))

        save_images(warp_cloth, name, os.path.join(opt.result_dir))

if __name__ == '__main__':
    opt = get_opt()
    test(opt)