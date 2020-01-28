from __future__ import print_function
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
#from models.networks_init import *
from dataloader_viton import *
import argparse
from tqdm import tqdm
# import torch.utils.data.distributed as DD
# from torch.nn.parallel import DistributedDataParallel as DDP
# from apex.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel as DP
# import horovod.torch as hvd
# import torch.distributed as dist
# from torch.multiprocessing import Process

from tensorboardX import SummaryWriter

from Lab.ClothNormalize import *

INPUT_SIZE = (192, 256)
EPOCHS = 30
PYRAMID_HEIGHT = 5

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_resize/train/")
    parser.add_argument("--datamode", default = "")
    parser.add_argument("--stage", default = "1")
    parser.add_argument("--data_list", default = "MVCup_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 1000)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    parser.add_argument("--smt_loss", type=float, default=1)
    parser.add_argument("--perc_loss", type=float, default=10.0)
    parser.add_argument("--struct_loss", type=float, default=10.0)
    parser.add_argument("--naming", type=str, default="default")

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
    
    model.train()
    Flow = FlowLoss().to(opt.gpu)

    L1 = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)


    

    writer = SummaryWriter(comment = "_" + opt.naming)

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

            transformed = model(con_cloth_mask, tar_cloth_mask)

            writer.add_image("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
            writer.add_image("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")
            writer.add_image("warp_cloth", transformed, cnt, dataformats="NCHW")
            
            loss = L1(transformed,tar_cloth_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            writer.add_scalar("loss", loss)
            #writer.add_scalar("loss/struct", struct, cnt)
            #writer.add_scalar("loss/smt", smt, cnt)
            #writer.add_scalar("loss/total", loss, cnt)
            writer.close()

            if (step + 1) % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.name, 'Epoch:%d_%05d.pth' % (epoch, (step+1))))
                model.to(opt.gpu)

if __name__ == '__main__':
    opt = get_opt()

    # create dataset 
    train_dataset = CPDataset(opt)

    # create dataloader
    train_loader = CPDataLoader(opt, train_dataset)
    
    model = ClothNormalizer()
    model = DP(model)
    train(opt,train_loader,model)