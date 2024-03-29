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
from Models.ClothNormalize_proj import *
from Models.networks import *
from dataloader_MVC import *

from canny import network as CNet
from canny import loss as CLoss

EPOCHS = 20
PYRAMID_HEIGHT = 6
NUM_STAGE = '2'

if IS_TOPS:
    stage = 'tops'
    nc = 2
    checkpoint = 'backup/init_top_5.pth'
    init_CN = 'backup/CN_top.pth'
else:
    stage = 'bottoms'
    nc = 2
    checkpoint = 'backup/init_bot_6.pth'
    init_CN = 'backup/CN_bot.pth'
dataroot = '/home/fashionteam/dataset_MVC_'+stage
dataroot_mask = '/home/fashionteam/ClothFlow/result/warped_mask/'+stage
datalist = 'train_MVC'+stage+'_pair.txt'
checkpoint_dir = osp.join(PWD,'stage'+NUM_STAGE,'checkpoints',stage)
runs = osp.join(PWD,'stage'+NUM_STAGE,'runs')
exp = 'train/'+stage

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "1")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", default = dataroot_mask)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = stage)
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 5)
    parser.add_argument("--grid_size", type=int, default = 10)
    parser.add_argument('--lr', type=float, default=0.00003, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoint_dir, help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default=checkpoint, help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 1)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--save_img_count", type=int, default = 50)
    parser.add_argument("--loss_count", type=int, default = 10)
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

def train(opt):
    model = FlowNet(PYRAMID_HEIGHT,4,1)
    model = nn.DataParallel(model)
    load_checkpoint(model, opt.checkpoint)
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    train_dataset = CFDataset(opt, is_tops=IS_TOPS)
    train_loader = CFDataLoader(opt, train_dataset)
    
    theta_generator = ClothNormalizer(nc=nc)
    load_checkpoint(theta_generator, init_CN)
    theta_generator.cuda()
    theta_generator.eval()

    Flow = FlowLoss(opt).cuda()

    writer = SummaryWriter(comment = "_" + opt.naming)

    # write options in text file
    if not os.path.exists("./options"): os.mkdir("./options")	
    f = open("./options/{}.txt".format(opt.naming), "w")
    temp_opt = vars(opt) 
    for key in temp_opt:
       val = temp_opt[key]
       f.write("{} --- {}\n".format(key, val))
    f.close()

    total_c = 0
    check_c = 0

    EdgeMaker = CNet.Canny()
    EdgeLoss = CLoss.cannyLoss()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1

            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs['cloth_mask'].cuda()
            tar_cloth = inputs['crop_cloth'].cuda()
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
            # pose = inputs['pose_png'].cuda()
            if IS_TOPS:
                c_pose = inputs['c_pose'].cuda()
                t_pose = inputs['t_pose'].cuda()

                # theta = theta_generator(c_pose,t_pose)
                theta = theta_generator(con_cloth_mask, tar_cloth_mask)
            else:
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

            
            if cnt > 500:
                edge_warped = EdgeMaker(warp_cloth)
                edge_tar = EdgeMaker(tar_cloth)
                WriteImage(writer,"edge_warped", edge_warped, cnt,dataformats="NCHW")
                WriteImage(writer,"edge_tar", edge_tar, cnt, dataformats="NCHW")
    
                edge_loss = EdgeLoss(edge_warped,edge_tar)

            loss, roi_perc, struct, smt, stat, abs = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth, con_cloth_mask)
            if cnt > 500:
                loss = loss + 0.01 * edge_loss[0]
            loss.backward()  
            if cnt % opt.loss_count == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step+1) % opt.save_img_count == 0 and opt.save_dir != "NONE":
               _dir = os.path.join(opt.save_dir, opt.naming, str(epoch)+"_"+str(step))
               if not os.path.exists(_dir): 
                   os.makedirs(_dir)
               warp_flow = F[0]
               numpy_warp_flow = warp_flow.data[0].detach().cpu().clone().numpy()
               np.save(os.path.join(_dir,'np.npy'), numpy_warp_flow)
               save_image(warp_cloth[0], os.path.join(_dir, "warp.png"))
               save_image(tar_cloth[0], os.path.join(_dir, "target.png"))
               save_image(con_cloth[0], os.path.join(_dir, "source.png"))

            if (step+1) % opt.display_count == 0:
                # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #     epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
                #     100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))
                writer.add_scalar("loss/roi_perc", roi_perc, cnt)
                writer.add_scalar("loss/struct", struct, cnt)
                writer.add_scalar("loss/smt", smt, cnt)
                writer.add_scalar("loss/total", loss, cnt)
                if cnt > 500:
                    writer.add_scalar("loss/canny_style", 0.01*edge_loss[0], cnt)
                writer.close()

            if (step+1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.naming, opt.stage, '%d_%05d.pth' % (epoch, (step+1))))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
    opt = get_opt()
    train(opt)
