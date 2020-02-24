from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from Models.networks import *
from dataloader_MVC import *
import argparse
from tqdm import tqdm
from torchvision.utils import save_image

from tensorboardX import SummaryWriter

EPOCHS = 10
PYRAMID_HEIGHT = 6
DATASET = 'MVC'
IS_TOPS = True

if DATASET is 'MVC':
    from dataloader_MVC import *
    stage = 'tops'
    nc = 2
    dataroot = '/home/fashionteam/dataset_MVC_'+stage
    dataroot_mask = '/home/fashionteam/ClothFlow/result/warped_mask/'+stage
    datalist = 'train_MVC'+stage+'_pair.txt'
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/init/'+stage
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
    parser.add_argument('-b', '--batch-size', type=int, default=4)
    
    parser.add_argument("--dataroot", default = dataroot)
    parser.add_argument("--dataroot_mask", default = dataroot_mask)
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "init")
    parser.add_argument("--data_list", default = datalist)
    parser.add_argument("--fine_width", type=int, default = INPUT_SIZE[0])
    parser.add_argument("--fine_height", type=int, default = INPUT_SIZE[1])
    parser.add_argument("--radius", type=int, default = 10)
    parser.add_argument("--grid_size", type=int, default = 10)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--exp", type=str, default = "initialize")

    parser.add_argument("--smt_loss", type=float, default=2)
    parser.add_argument("--perc_loss", type=float, default=1)
    parser.add_argument("--struct_loss", type=float, default=10)
    parser.add_argument("--stat_loss", type=float, default=0) 
    parser.add_argument("--abs_loss", type=float, default=0)
    parser.add_argument("--init_name", type=str, default = "")
    parser.add_argument("--naming", type=str, default="init")
    parser.add_argument("--save_dir", type=str, default="npz")

    opt = parser.parse_args()
    return opt

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("error")
        exit()
    model.load_state_dict(torch.load(checkpoint_path), False)
    model.cuda()

def train(opt):

    ####
    A = np.array([[[1,0,0],[0,1,0]]]).astype(np.float32)
    A = np.concatenate([A]*opt.batch_size,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A,(opt.batch_size,2,INPUT_SIZE[1],INPUT_SIZE[0])).cuda()
    ####

    model = FlowNet(PYRAMID_HEIGHT,4,2)
    model = nn.DataParallel(model)
    # load_checkpoint(model,"backup/initial__4_02426.pth")##
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)

    Flow = nn.L1Loss()

    writer = SummaryWriter("MSruns/initial")

    for epoch in tqdm(range(EPOCHS), desc='EPOCH'):
        for step in tqdm(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs['cloth_mask'].cuda()
            tar_cloth = inputs['crop_cloth'].cuda()
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
            pose = inputs['pose_png'].cuda()

            writer.add_image("con_cloth", con_cloth, cnt, dataformats="NCHW")
            writer.add_image("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
            writer.add_image("tar_cloth", tar_cloth, cnt, dataformats="NCHW")
            writer.add_image("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")
            writer.add_image("pose", pose, cnt, dataformats="NCHW")

            [F, warp_cloth, warp_mask] = model(torch.cat([con_cloth, con_cloth_mask], 1), torch.cat([tar_cloth_mask, pose], 1))

            writer.add_image("warp_cloth", warp_cloth, cnt, dataformats="NCHW")
            writer.add_image("warp_mask", warp_mask, cnt, dataformats="NCHW")

            optimizer.zero_grad()
            #loss, roi_perc, struct, smt = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth)
            loss = Flow(F[0].transpose(1,2).transpose(2,3),net)###
            loss.backward()
            optimizer.step()

            # if (step+1) % opt.display_count == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
            #         100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))
            writer.add_scalar("loss/identity_loss",loss,cnt)
            writer.close()
               
            if step % opt.save_count == 0 and opt.save_dir != "NONE":
               _dir = os.path.join(opt.save_dir, opt.naming, str(epoch)+"_"+str(step))
               if not os.path.exists(_dir): 
                   os.makedirs(_dir)
               warp_flow = F[-1]
               numpy_warp_flow = warp_flow.data[0].detach().cpu().clone().numpy()
               np.save(os.path.join(_dir,'np.npy'), numpy_warp_flow)
               save_image(warp_cloth[0], os.path.join(_dir, "warp.png"))
               save_image(tar_cloth[0], os.path.join(_dir, "target.png"))
               save_image(con_cloth[0], os.path.join(_dir, "source.png"))

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.stage, 'initial_%s_%d_%05d.pth' % (opt.init_name,epoch, (step+1))))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
    opt = get_opt()
    train(opt)
