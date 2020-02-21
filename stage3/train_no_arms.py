from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as Ftnl
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from Models.UNetS3_no_arms import *
from Models.LossS3_debug import *
from dataloader_viton_full import *
import argparse
from tqdm import tqdm_notebook
from torchvision.utils import save_image

from tensorboardX import SummaryWriters
import sys
from Lab.ClothNormalize import ClothNormalizer
import time
import pickle

EPOCHS = 18
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
    exp = 'train/'+stage
else:
    from dataloader_viton import *
    dataroot = '/home/fashionteam/viton_resize/'
    datalist = ''
    checkpoint_dir = '/home/fashionteam/ClothFlow/stage2/checkpoints/tops/'
    exp = 'train/tops/'
   


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_resize/")
    parser.add_argument("--warped_cloth_path", type=str, default=osp.join(PWD,"result/warped_cloth"))
    parser.add_argument("--warped_mask_path", type=str, default=osp.join(PWD,"result/warped_mask"))
    parser.add_argument("--datamode", default = "train")
    parser.add_argument("--stage", default = "stage3")
    parser.add_argument("--data_list", default = "MVCup_pair.txt")
    parser.add_argument("--fine_width", type=int, default = 192)
    parser.add_argument("--fine_height", type=int, default = 256)
    parser.add_argument("--radius", type=int, default = 3)
    parser.add_argument("--grid_size", type=int, default = 5)
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='/home/fashionteam/ClothFlow/stage3/checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default='result', help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    
    parser.add_argument("--smt_loss", type=float, default=2)
    parser.add_argument("--perc_loss", type=float, default=1)
    parser.add_argument("--struct_loss", type=float, default=1.7*10)
    parser.add_argument("--stat_loss", type=float, default=0)
    parser.add_argument("--abs_loss", type=float, default=0)
    #parser.add_argument("--naming", type=str, default="default")
    parser.add_argument("--save_dir", type=str, default="/home/fashionteam/ClothFlow/npz")
    parser.add_argument("--exp", type=str, default="head_init")
    

    opt = parser.parse_args()
    return opt

def save_checkpoint(model, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    torch.save(model.cpu().state_dict(), save_path)
    model.cuda()

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(1/0)
        print("ERROR")
        return 
    model.load_state_dict(torch.load(checkpoint_path))	
    model.cuda()

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

def WriteImage(writer,name,data,cnt):
    data_ = (data.clone() + 1)*0.5
    #data_ = data_.cpu().clamp(0,255).detach().numpy().astype('uint8')
    #data_ = data_.swapaxes(1,2).swapaxes(2,3)
    #print(data_.shape)
    writer.add_images(name,data_,cnt)

def train(opt):

    model = UNet(opt)
    model = nn.DataParallel(model,output_device=0)
    #load_checkpoint(model, "/home/fashionteam/ClothFlow/stage3/checkpoints/experiment/5_00350.pth")
    #load_checkpoint(model, "/home/fashionteam/ClothFlow/checkpoints/default/stage2/2_00240.pth")
    model.cuda()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    train_dataset = CFDataset(opt)
    train_loader = CFDataLoader(opt, train_dataset)

    #theta_generator = ClothNormalizer()
    #load_checkpoint(theta_generator,"/home/fashionteam/ClothFlow/Lab/saved/G_theta_affine.pth")
    #theta_generator.cuda()
    #theta_generator.eval()
    
    #Flow = FlowLoss(opt).cuda()

    writer = SummaryWriter("/home/fashionteam/ClothFlow/stage3/debug/runs/"+opt.exp)
    rLoss = renderLoss()

    

    for epoch in tqdm_notebook(range(EPOCHS), desc='EPOCH'):
        for step in tqdm_notebook(range(len(train_loader.dataset)//opt.batch_size + 1), desc='step'):
            cnt = epoch * (len(train_loader.dataset)//opt.batch_size + 1) + step + 1
            
            inputs = train_loader.next_batch()

            con_cloth = inputs['cloth'].cuda()
            #tar_cloth = inputs['crop_cloth'].cuda()
            #tar_cloth_mask = inputs['crop_cloth_mask'].cuda()
            warped = inputs['warped'].cuda()
            answer = inputs['image'].cuda()
            off_cloth = inputs['off_cloth'].cuda()
            pose = inputs['pose'].cuda()
            #shape = inputs['shape'].cuda()
            #B_,H_,W_ = shape.shape
            #shape = shape.view(B_,1,H_,W_)
            head = inputs['head'].cuda()
            #arms = inputs['crop_arms'].cuda()
            pants = inputs['crop_pants'].cuda()
            name = inputs['name']

            result = model(con_cloth,off_cloth,pose,warped,head,pants)

            if cnt % 50 == 0:
                WriteImage(writer,"GT", answer, cnt)
                #WriteImage(writer,"shape", shape, cnt)
                WriteImage(writer,"warped", warped, cnt)
                WriteImage(writer,"con_cloth", con_cloth, cnt)#, dataformats="NCHW")
                WriteImage(writer,"off_cloth", off_cloth, cnt)
                WriteImage(writer,"Head", head, cnt)
                WriteImage(writer,"Result", result, cnt)
                
            #save_images(result,name,"what")
            
            if epoch == 0:
                optimizer.zero_grad()
                loss, percept, style = rLoss(result,head)
                loss.backward()
                optimizer.step()
            
            else:	
                optimizer.zero_grad()
                loss, percept, style = rLoss(result,answer)
                loss.backward()
                optimizer.step()

            if (step+1) % opt.display_count == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch+1, (step+1) * 1, len(train_loader.dataset)//opt.batch_size + 1,
                    100. * (step+1) / (len(train_loader.dataset)//opt.batch_size + 1), loss.item()))
            writer.add_scalar("loss/loss", loss, cnt)
            writer.add_scalar("loss/percept", percept, cnt)
            writer.add_scalar("loss/style", style, cnt)

            writer.close()

            if cnt % opt.save_count == 0:
                save_checkpoint(model, os.path.join(opt.checkpoint_dir, opt.exp, '%d_%05d.pth' % (epoch, (step+1))))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "3"

    opt = get_opt()
    #with open("opt.pkl","wb") as f:
    #    pickle.dump(opt,f)
    train(opt)
