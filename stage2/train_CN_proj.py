from __future__ import print_function
import sys
import torch
import torch.nn as nn
import torch.nn.functional as Ftnl
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
#from models.networks_init import *
from dataloader_viton import *
import argparse
from tqdm import tqdm
import math

from torchvision.utils import save_image
from torch.nn import DataParallel as DP

from tensorboardX import SummaryWriter

from Models.ClothNormalize_proj import *

EPOCHS = 30
PYRAMID_HEIGHT = 5
DATASET = 'MVC'
IS_TOPS = True

if DATASET is 'MVC':
    from dataloader_MVC import *
    if IS_TOPS:
        stage = 'tops'
    else:
        stage = 'bottoms'
    dataroot = '/home/fashionteam/dataset_MVC_'+stage
    dataroot_mask = '/home/fashionteam/ClothFlow/result/warped_mask/'+stage
    datalist = 'MVC'+stage+'_pair.txt'
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/'+stage
    exp = 'train/'+stage
else:
    from dataloader_viton import *
    dataroot = '/home/fashionteam/viton_resize/'
    datalist = ''
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/tops/'
    exp = 'train/tops/'

def get_A(bs, H, W):
    A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
    A = np.concatenate([A]*bs,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A,(bs,2,INPUT_SIZE[1],INPUT_SIZE[0])).cuda()
    return net

def projection_grid(param, shape):
    alpha = torch.unsqueeze(torch.unsqueeze(param[:,0:1],2),3)
    beta = torch.unsqueeze(torch.unsqueeze(param[:,1:2],2),3)
    scale = torch.add(torch.unsqueeze(torch.unsqueeze(param[:,2:3],2),3), 1)
    theta = torch.unsqueeze(torch.unsqueeze(param[:,3:4],2),3)
    dis = 3
    # dis = torch.unsqueeze(torch.unsqueeze(param[:,4:],2),3)
    base = get_A(shape[0], shape[2], shape[3])
    base_X = base[:,:,:,0:1]
    base_Y = base[:,:,:,1:]
    # denominator = base_X * theta - dis * torch.sqrt(1 - theta ** 2)
    denominator = base_X * torch.sin(theta) - dis * torch.cos(theta)
    X = - 1 / scale * (dis * base_X / denominator + alpha)
    # Y = (dis * torch.sqrt(1 - theta ** 2)) / (-denominator * scale) * base_Y - beta / scale
    Y = (dis * torch.cos(theta)) / (-denominator * scale) * base_Y - beta / scale
    grid = torch.cat([X,Y], axis=3)
    return grid 

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", default = dataroot_mask)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "1")
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--save_img_count", type=int, default = 10)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--exp", default = "ClothNormalizer_proj_bot")
    
    parser.add_argument("--save_dir", type=str, default="npz")
    parser.add_argument("--naming", type=str, default="ClothNomalizer")

    opt = parser.parse_args()
    print(opt)
    return opt

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except:
            pass

    torch.save(model.cpu().state_dict(), save_path)


def train(opt,train_loader,model):


    model.cuda()
    model.train()
    

    L1 = nn.L1Loss()
    L2 = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    
    train_dataset = CFDataset(opt, is_tops=IS_TOPS)
    train_loader = CFDataLoader(opt, train_dataset)

    runs = "./runs/"
    if os.path.isdir(os.path.join(runs, opt.exp)):
        os.system('rm -r '+runs+opt.exp)
    os.makedirs(runs+opt.exp)
    writer = SummaryWriter(runs+opt.exp)

    # write options in text file
    if not os.path.exists("./options"): os.mkdir("./options")	
    f = open("./options/{}.txt".format(opt.naming), "w")
    temp_opt = vars(opt)
    for key in temp_opt:
       val = temp_opt[key]
       f.write("{} --- {}\n".format(key, val))
    f.close()

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            inputs = train_loader.next_batch()

            # c_pose = inputs['c_pose'].cuda()
            # t_pose = inputs['t_pose'].cuda()
            con_cloth_mask = inputs['cloth_mask'].cuda()
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
            con_cloth = inputs['cloth'].cuda()
            tar_cloth = inputs['crop_cloth'].cuda()
            t_name = inputs['t_name']

            param = model(con_cloth_mask, tar_cloth_mask)
            print(param.shape, con_cloth_mask.shape)
            print(param)
            grid = projection_grid(param, tar_cloth_mask.shape)
            batch, H, W, C = grid.shape
            AAA = get_A(batch,H,W)
            print(grid.shape)
            transformed = Ftnl.grid_sample(con_cloth_mask, grid)
            transformed_cloth = Ftnl.grid_sample(con_cloth, grid)

            if (step+1) % opt.save_img_count == 0:
                writer.add_image("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
                writer.add_image("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")
                writer.add_image("warp_cloth_mask", transformed, cnt, dataformats="NCHW")

                writer.add_image("con_cloth", con_cloth, cnt, dataformats="NCHW")
                writer.add_image("tar_cloth", tar_cloth, cnt, dataformats="NCHW")
                writer.add_image("warp_cloth", transformed_cloth, cnt, dataformats="NCHW")
            ####

            #print(grid_loss)
            # det_loss
            # LU = theta[:,0,0]
            # LD = theta[:,1,0]
            # RU = theta[:,0,1]
            # RD = theta[:,1,1]
            # det = LU*RD - LD*RU
            # det_loss = torch.mean(Ftnl.relu(-det))
            # writer.add_scalar("loss/det_loss", det_loss,cnt)


            ####

            rotate = np.array(t_name)
            rotate = np.where(rotate=='3', 0.8, -0.8)
            rotate = torch.unsqueeze(torch.from_numpy(rotate),1).cuda()
            print(rotate)
            loss = 0

            if cnt < 100:
                grid_loss = (grid-AAA)**2
                grid_loss = torch.mean(torch.sqrt(torch.sum(grid_loss,axis=3)))
                writer.add_scalar("loss/grid_loss", grid_loss,cnt)
                initial_loss = L1(grid_loss,AAA)
                writer.add_scalar("loss/initial_loss", initial_loss,cnt)
                scale_loss = L1(param[:, 2:3], torch.zeros(param[:, 2:3].shape).cuda()) * 10
                writer.add_scalar("loss/scale_loss", scale_loss,cnt)    
                # theta_loss = L1(param[:, 3:], torch.zeros(param[:, 3:].shape).cuda()) * 10
                # writer.add_scalar("loss/theta_loss", theta_loss,cnt)
                loss = loss + initial_loss + scale_loss #+ theta_loss

            if cnt >= 100:
                rotate_loss = L1(param[:, 3:], rotate)
                writer.add_scalar("loss/rotate_loss", rotate_loss,cnt)
                loss = loss + rotate_loss

            if cnt >= 400:
                original_loss = torch.mean(torch.abs(Ftnl.leaky_relu(tar_cloth_mask-transformed,0.8))) * 10
                writer.add_scalar("loss/original_loss", original_loss,cnt)
                loss = loss + original_loss #+ det_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("loss/loss", loss,cnt)
            
            if (step+1) % opt.save_img_count == 0 and opt.save_dir != "NONE":
               _dir = os.path.join(opt.save_dir, opt.naming, str(epoch)+"_"+str(step))
               if not os.path.exists(_dir): 
                   os.makedirs(_dir)
               warp_flow = grid.transpose(2,3).transpose(1,2)
               numpy_warp_flow = warp_flow.data[0].detach().cpu().clone().numpy()
               np.save(os.path.join(_dir,'np.npy'), numpy_warp_flow)
               save_image(transformed[0], os.path.join(_dir, "warp.png"))
               save_image(tar_cloth_mask[0], os.path.join(_dir, "target.png"))
               save_image(con_cloth_mask[0], os.path.join(_dir, "source.png"))

            writer.close()

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.exp, 'Epoch:%d_%05d.pth' % (epoch, (step+1))))
                model.cuda()

if __name__ == '__main__':
    opt = get_opt()

    # create dataset 
    train_dataset = CFDataset(opt)

    # create dataloader
    train_loader = CFDataLoader(opt, train_dataset)
    
    model = ClothNormalizer(depth=PYRAMID_HEIGHT, nc=2)
    #model = DP(model)
    train(opt,train_loader,model)