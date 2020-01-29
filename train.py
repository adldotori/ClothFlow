from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models.networks_a import *
from dataloader_viton import *
import argparse
from tqdm import tqdm_notebook
from torchvision.utils import save_image

from tensorboardX import SummaryWriter

INPUT_SIZE = (192, 256)
EPOCHS = 15
PYRAMID_HEIGHT = 4



def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_resize/train/")
    parser.add_argument("--datamode", default = "")
    parser.add_argument("--stage", default = "1")
    parser.add_argument("--data_list", default = "MVCup_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    parser.add_argument("--smt_loss", type=float, default=2)
    parser.add_argument("--perc_loss", type=float, default=1)
    parser.add_argument("--struct_loss", type=float, default=10)
    parser.add_argument("--naming", type=str, default="default")
    parser.add_argument("--save_dir", type=str, default="NONE")

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

    A = np.array([[[1, 0, 0], [0, 1, 0]]]).astype(np.float32)
    A = np.concatenate([A]*8,0)
    A = torch.from_numpy(A)
    net = nn.functional.affine_grid(A, (8, 2, 256, 192)).cuda()

    model = FlowNet(PYRAMID_HEIGHT, 4, 1)
    model = nn.DataParallel(model,output_device=0)
# load_checkpoint(model, "./backup/initial_with_smoothing_0_00400.pth")
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)
    
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

    for epoch in tqdm_notebook(range(EPOCHS), desc='EPOCH'):
        for step in tqdm_notebook(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            con_cloth_mask = inputs['cloth_mask'].cuda()
            tar_cloth = inputs['crop_cloth'].cuda()
            tar_cloth_mask = inputs['crop_cloth_mask'].cuda()

            writer.add_images("con_cloth", con_cloth, cnt)
            writer.add_images("con_cloth_mask", con_cloth_mask, cnt, dataformats="NCHW")
            writer.add_images("tar_cloth", tar_cloth, cnt)
            writer.add_images("tar_cloth_mask", tar_cloth_mask, cnt, dataformats="NCHW")

            [F, warp_cloth, warp_mask, warp_list] = model(torch.cat([con_cloth, con_cloth_mask], 1), tar_cloth_mask)

            writer.add_images("warp_cloth", warp_cloth, cnt)
            writer.add_images("warp_mask", warp_mask, cnt, dataformats="NCHW")
            writer.add_images("warp_first", warp_list[1][:, :3, :, :], cnt)
            
            if step % opt.save_count == 0 and opt.save_dir != "NONE":
               _dir = os.path.join(opt.save_dir, opt.naming, str(epoch)+"_"+str(step))
               if not os.path.exists(_dir): 
                   os.makedirs(_dir)
               warp_flow = F[-1]
               numpy_warp_flow = warp_flow.data[0].detach().cpu().clone().numpy()
               np.save(_dir, numpy_warp_flow)
               save_image(warp_cloth[0], os.path.join(_dir, "warp.png"))
               save_image(tar_cloth[0], os.path.join(_dir, "target.png"))
               save_image(con_cloth[0], os.path.join(_dir, "source.png"))
					
            optimizer.zero_grad()
            loss, roi_perc, struct, smt = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth, warp_list)
            loss.backward()
            optimizer.step()

            if (step+1) % opt.display_count == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
                    100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))
            writer.add_scalar("loss/roi_perc", roi_perc, cnt)
            writer.add_scalar("loss/struct", struct, cnt)
            writer.add_scalar("loss/smt", smt, cnt)
            writer.add_scalar("loss/total", loss, cnt)
            writer.close()

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.naming, opt.stage, '%d_%05d.pth' % (epoch, (step+1))))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "2"

    opt = get_opt()
    train(opt)
