from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as Ftnl
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from Models.UNetS3 import *
from Models.LossS3 import *
from dataloader_MVC import *
import argparse
from tqdm import tqdm
from torchvision.utils import save_image
import os.path as osp

from tensorboardX import SummaryWriter
import sys
import time

EPOCHS = 15
PYRAMID_HEIGHT = 4
IS_TOPS = False # TOPS or BOTTOMS 

if IS_TOPS:
    dataroot = '/home/fashionteam/dataset_MVC_tops/'
    datalist = 'MVCtops_pair.txt'
    exp = 'test/tops/'
    result_dir = '/home/fashionteam/ClothFlow/result/warped_mask/tops/'
    in_channels = 22
else:
    dataroot = '/home/fashionteam/dataset_MVC_bottoms/'
    datalist = 'MVCbottoms_pair.txt'
    exp = 'test/bottoms/'
    result_dir = '/home/fashionteam/ClothFlow/result/warped_mask/bottoms/'
    in_channels = 2

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    
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
    parser.add_argument('--checkpoint_dir', type=str, default='/home/fashionteam/ClothFlow/stage1/checkpoints', help='save checkpoint infos')
    parser.add_argument('--result_dir', type=str, default=result_dir, help='save result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 500)
    parser.add_argument("--save_count_npz", type=int, default = 50)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--exp", type=str, default=exp)
    

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

def test(opt):

    model = UNet(opt, in_channels=in_channels)
    model = nn.DataParallel(model,output_device=0)
    load_checkpoint(model, "../backup/stage1_bot_512.pth")
    model.cuda()
    model.train()
    test_dataset = CFDataset(opt, is_tops=IS_TOPS)
    test_loader = CFDataLoader(opt, test_dataset)

    #theta_generator = ClothNormalizer()
    #load_checkpoint(theta_generator,"/home/fashionteam/ClothFlow/Lab/saved/G_theta_affine.pth")
    #theta_generator.cuda()
    #theta_generator.eval()
    
    #Flow = FlowLoss(opt).cuda()

    writer = SummaryWriter("/home/fashionteam/ClothFlow/stage1/runs/"+opt.exp)
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
            result = model(con_cloth, con_cloth_mask, pose)
        else:
            result = model(con_cloth_mask, tar_body_mask, None)

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

        save_images(result, name, opt.result_dir)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]= "0,1,2,3"

    opt = get_opt()
    test(opt)
