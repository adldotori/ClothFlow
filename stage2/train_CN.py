from __future__ import print_function
import sys
sys.path.append('..')
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
# import torch.utils.data.distributed as DD
# from torch.nn.parallel import DistributedDataParallel as DDP
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
# import horovod.torch as hvd
# import torch.distributed as dist
# from torch.multiprocessing import Process

from tensorboardX import SummaryWriter

from Models.ClothNormalize import *

EPOCHS = 30
PYRAMID_HEIGHT = 5

def get_A(bs, H, W):
    A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
    A = np.concatenate([A]*bs,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A,(bs,2,INPUT_SIZE[1],INPUT_SIZE[0])).cuda()
    return net

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_resize/")
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "1")
    parser.add_argument("--data_list", default = "MVCup_pair_unorder.txt")
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--exp", default = "ClothNormalizer_5")
    

    
    parser.add_argument("--smt_loss", type=float, default=1)
    parser.add_argument("--perc_loss", type=float, default=10.0)
    parser.add_argument("--struct_loss", type=float, default=10.0)
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

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)


    

    writer = SummaryWriter("runs/"+opt.exp)

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

            #con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs['cloth_mask'].cuda()
            #tar_cloth = inputs['crop_cloth'].cuda()
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()

            theta = model(con_cloth_mask, tar_cloth_mask)
            grid = Ftnl.affine_grid(theta, tar_cloth_mask.shape)
            batch, H, W, C = grid.shape
            AAA = get_A(batch,H,W)
            transformed = Ftnl.grid_sample(con_cloth_mask , grid)#,padding_mode="border")#,q,torch.mean(tar_mask)

            #transformed = model(con_cloth_mask, tar_cloth_mask)

            writer.add_image("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
            writer.add_image("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")
            writer.add_image("warp_cloth", transformed, cnt, dataformats="NCHW")

            ####
            grid_loss = (grid-AAA)**2 
            grid_loss = torch.mean(torch.sqrt(torch.sum(grid_loss,axis=3)))
            writer.add_scalar("loss/grid_loss", grid_loss,cnt)
            #print(grid_loss)
            # det_loss
            LU = theta[:,0,0]
            LD = theta[:,1,0]
            RU = theta[:,0,1]
            RD = theta[:,1,1]
            det = LU*RD - LD*RU
            det_loss = torch.mean(Ftnl.relu(-det))
            writer.add_scalar("loss/det_loss", det_loss,cnt)


            ####
            original_loss = torch.mean(torch.abs(Ftnl.leaky_relu(tar_cloth_mask-transformed,0.1764)))
            writer.add_scalar("loss/original_loss", original_loss,cnt)

            initial_loss = L1(grid_loss,AAA)
            writer.add_scalar("loss/initial_loss", initial_loss,cnt)
            if cnt < 50:
                loss = initial_loss
            else:
                loss = original_loss+ det_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("loss/loss", loss,cnt)
            
            #writer.add_scalar("loss/q", q[0],cnt)
            #writer.add_scalar("loss/ratio", ratio,cnt)
            #writer.add_scalar("loss/struct", struct, cnt)
            #writer.add_scalar("loss/smt", smt, cnt)
            #writer.add_scalar("loss/total", loss, cnt)
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
    
    model = ClothNormalizer(depth=PYRAMID_HEIGHT)
    #model = DP(model)
    train(opt,train_loader,model)