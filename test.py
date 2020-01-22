from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from models.networks import *
from dataloader_viton import *
import argparse

INPUT_SIZE = (192, 256)
PYRAMID_HEIGHT = 4

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default = "TryOn")
    parser.add_argument("--gpu_ids", default = "0")
    parser.add_argument('-j', '--workers', type=int, default=1)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    
    parser.add_argument("--dataroot", default = "/home/fashionteam/viton_resize/test/")
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
    parser.add_argument('--test', type=str, default='test', help='save test result infos')
    parser.add_argument('--checkpoint', type=str, default='', help='model checkpoint for initialization')
    parser.add_argument("--display_count", type=int, default = 20)
    parser.add_argument("--save_count", type=int, default = 100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')

    opt = parser.parse_args()
    return opt

def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print('[-] Checkpoint Load Error!')        
        exit()
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)

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
    model = FlowNet(PYRAMID_HEIGHT, 4, 1)
    load_checkpoint(model, opt.checkpoint)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    test_dataset = CFDataset(opt)
    test_loader = CFDataLoader(opt, test_dataset)

    model.to(device)
    model.eval()
    Flow = FlowLoss().to(device)

    for step in range(len(test_loader.dataset)):
        inputs = test_loader.next_batch()
        name = inputs['name']
        con_cloth = inputs['cloth'].to(device)
        con_cloth_mask = inputs['cloth_mask'].to(device)
        tar_cloth = inputs['crop_cloth'].to(device) 
        tar_cloth_mask = inputs['crop_cloth_mask'].to(device)

        [F, warp_cloth, warp_mask] = model(torch.cat([con_cloth, con_cloth_mask], 1), tar_cloth_mask)
        # optimizer.zero_grad()
        loss, roi_perc, struct, smt = Flow(PYRAMID_HEIGHT, F, warp_mask, warp_cloth, tar_cloth_mask, tar_cloth)
        # loss.backward()
        # optimizer.step()

        if (step+1) % opt.display_count == 0:
            print('Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                 (step+1) * 1, len(test_loader.dataset),
                100. * (step+1) / len(test_loader.dataset), loss.item()))

        save_images(warp_mask, name, os.path.join(opt.test))

if __name__ == '__main__':
    if not os.path.isdir('test'):
        os.system('mkdir test')

    opt = get_opt()
    test(opt)