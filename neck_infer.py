import sys
sys.path.append("makeneck")
from utils import *
from makeneck.Models import UNetS3 as M
import os.path as osp

import argparse
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from PIL import Image
from PIL import ImageDraw
from pathlib import Path
import numpy as np
import json
import pickle
import neckmake
import torch.nn.functional as Ftnl
import cv2

def imsave(result,path):
    img = (result[0].clone()+1)*0.5 * 255
    if img.shape[0] == 1:
        img = img[0,:,:]
    else:
        img = img.transpose(0,1).transpose(1,2)
    img = img.cpu().clamp(0,255)
    img = img.detach().numpy().astype('uint8')
    Image.fromarray(img).save(path)

def masksave(result, path):
	img = (result[0].clone()) * 255
	if img.shape[0] == 1:
		img = img[0, :, :]
	else: 
		img = img.transpose(0, 1).transpose(1, 2)
	img = img.cpu()#.clamp(0, 255)
	img = img.detach().numpy().astype('uint8')
	Image.fromarray(img).save(path)


def get_opt():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/home/fashionteam/underwear_512")
    parser.add_argument("--mode", default = "one") # mode = all | one
    parser.add_argument("--name", default = "necktest") # valid if mode == one
    parser.add_argument("--version", default = "viton") # version = MVC | vitons
    parser.add_argument("--is_top", default = True) # valid if version == MVC
    parser.add_argument("--PYRAMID_HEIGHT", default = 5)
    # parser.add_argument("--stage1_model_pth", default = "stage1/checkpoints/tops/checkpoint_1107.pth")
    parser.add_argument("--makeneck_model_pth", default = "stage3/checkpoints/checkpoint_72000.pth")
    parser.add_argument("--result", default = "test_uw")
    parser.add_argument("--height", default = 512)
    parser.add_argument("--width", default = 512)

    opt = parser.parse_args()
    return opt

def load_inputs(opt,name):
    H = opt.height
    W = opt.width
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    data_path = osp.join(opt.dataroot,name)
    path_image = osp.join(data_path,"image.png")
    path_segment = osp.join(data_path,"segment.png")
    path_pose_pkl = osp.join(data_path,"pose.pkl")
    path_noneck = osp.join(data_path, "nonneck.jpg")
    path_noneck_mask = osp.join(data_path, "image_mask.jpg")
    image = Image.open(path_image).convert('RGB')

    image = transform(image)

    seg = Image.open(path_segment)
    parse = transform_1ch(seg)
    parse_array = np.array(seg)
    
    mask = Image.open(path_noneck_mask)
    mask = transform_1ch(mask)
    mask_array = np.array(mask)

    shape = (parse_array > 0).astype(np.float32)  # condition body shape
    head = (parse_array == 1).astype(np.float32) + \
                (parse_array == 2).astype(np.float32) + \
                (parse_array == 4).astype(np.float32) + \
                (parse_array == 13).astype(np.float32)
    cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)
    arms = (parse_array == 15).astype(np.float32) + \
                (parse_array == 14).astype(np.float32)
    pants = (parse_array == 9).astype(np.float32) + \
                (parse_array == 12).astype(np.float32)
    
    mask = (mask_array > 0).astype(np.float32)
    mask = torch.from_numpy(mask)
    # nonneck = image * shape
    nonneck = transform(Image.open(path_noneck).convert('RGB')) * mask
    print(mask.shape)

    head = torch.from_numpy(head)
    crop_head = image * head + (1 - head)

    with open(path_pose_pkl, 'rb') as f:
        pose_label = pickle.load(f)
    pose_data = pose_label
    
    point_num = 18
    pose_map = torch.zeros(point_num, H, W)
    r = 5
    pose = Image.new('L', (W, H))
    pose_draw = ImageDraw.Draw(pose)
    for i in range(point_num):
    # for i in [0,1,2,3,4,5,6,7,8,11,14,15,16,17]:
        one_map = Image.new('L', (W, H))
        draw = ImageDraw.Draw(one_map)
        if i in pose_data.keys():
            pointx = pose_data[i][0]
            pointy = pose_data[i][1]
        else:
            pointx = -1
            pointy = -1

        #c_pointx = c_pointx * 192 / 762
        #c_pointy = c_pointy * 256 / 1000
        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
        one_map = transform_1ch(one_map)
        pose_map[i] = one_map[0]
    
    results = {
        'image': image,  # source image
        'nonneck' : nonneck,
        'parse': parse,
        'pose': pose_map,
        'head': crop_head,
    }
    return results
    



def main():
    opt = get_opt()
    PYRAMID_HEIGHT = opt.PYRAMID_HEIGHT

    makeneck = M.UNet(opt,22,5)
    makeneck = nn.DataParallel(makeneck)
    makeneck.eval()

    load_checkpoint(makeneck,opt.makeneck_model_pth)

    set_requires_grad(makeneck,False)

    if opt.mode == "all":
        names = os.listdir(opt.dataroot)
    elif opt.mode == "one":
        names = [opt.name]
    else:
        print("--mode should be all or one")
        assert(1==0)
    
    for name in names:
        inputs = load_inputs(opt,name)
        nonneck = inputs['nonneck'].cuda()
        answer = inputs['image'].cuda()
        parse = inputs['parse'].cuda()
        pose = inputs['pose'].cuda()
        head = inputs['head'].cuda()

        pose.unsqueeze_(0)
        nonneck.unsqueeze_(0)
        parse.unsqueeze_(0)
        
        if not os.path.exists(osp.join(opt.result, name)): os.makedirs(osp.join(opt.result,name))
        
        result = makeneck(pose, nonneck, parse)
        imsave(nonneck, osp.join(opt.result,name,"nonneck.jpg")) 
        imsave(result, osp.join(opt.result,name,"makeneck.jpg"))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    #torch.cuda.set_device(2)
    main()
