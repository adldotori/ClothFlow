import sys
sys.path.append("stage1")
sys.path.append("stage2")
sys.path.append("stage3")
from utils import *
from stage1.Models import UNetS3 as M1
from stage2.Models import networks as M2
from stage3.Models import UNetS3 as M3
from stage2.Models import ClothNormalize_proj as CN
from stage2 import train_CN_proj as projection
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
from stage0 import neckmake
import torch.nn.functional as Ftnl

def imsave(result,path):
    img = (result[0].clone()+1)*0.5 * 255
    if img.shape[0] == 1:
        img = img[0,:,:]
    else:
        img = img.transpose(0,1).transpose(1,2)
    img = img.cpu().clamp(0,255)
    img = img.detach().numpy().astype('uint8')
    Image.fromarray(img).save(path)

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/home/fashionteam/final_dataset_white")
    parser.add_argument("--mode", default = "one") # mode = all | one
    parser.add_argument("--name", default = "raw_1") # valid if mode == one
    parser.add_argument("--version", default = "MVC") # version = MVC | vitons
    parser.add_argument("--is_top", default = True) # valid if version == MVC
    parser.add_argument("--PYRAMID_HEIGHT", default = 5)
    parser.add_argument("--stage1_model_pth", default = "backup/stage1_top_512.pth")
    parser.add_argument("--stage2_model_pth", default = "backup/stage2_top_512.pth")
    parser.add_argument("--stage3_model_pth", default = "backup/stage3_top_512.pth")
    parser.add_argument("--clothnormalize_model_pth", default = "backup/CN_top_.pth")
    parser.add_argument("--result", default = "test6")
    parser.add_argument("--height", default = 512)
    parser.add_argument("--width", default = 512)

    opt = parser.parse_args()
    print(opt)
    return opt

def load_inputs(opt,name):
    H = opt.height
    W = opt.width
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    data_path = osp.join(opt.dataroot,name)
    path_cloth = osp.join(data_path,"cloth.png")
    path_image = osp.join(data_path,"image.jpg")
    path_cloth_mask = osp.join(data_path,"cloth_mask.png")
    path_segment = osp.join(data_path,"segment.png")
    path_pose_pkl = osp.join(data_path,"pose.pkl")
    if opt.version == "MVC":
        path_pose_png = osp.join(data_path,"pose.png")
        path_segment_vis = osp.join(data_path,"segment_vis.png")

    cloth = Image.open(path_cloth).convert('RGB')
    image = Image.open(path_image).convert('RGB')
    cloth_mask = Image.open(path_cloth_mask)

    c_mask_array = np.array(cloth_mask)
    c_mask_array = (c_mask_array > 0).astype(np.float32)
    c_mask = torch.from_numpy(c_mask_array)
    cloth_mask = c_mask.unsqueeze_(0)
    cloth_ = transform(cloth)
    image = transform(image)

    seg = Image.open(path_segment)
    parse_array = np.array(seg)
    parse_fla = parse_array.reshape(H*W, 1)
    parse_fla = torch.from_numpy(parse_fla).long()
    parse = torch.zeros(H*W, 20).scatter_(1, parse_fla, 1)
    parse = parse.view(H, W, 20)
    parse = parse.transpose(2, 0).transpose(2,1).contiguous()  # [20,256,192]

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
    
    head = torch.from_numpy(head)
    cloth = torch.from_numpy(cloth)
    arms = torch.from_numpy(arms)
    pants = torch.from_numpy(pants)

    crop_head = image * head + (1 - head)
    crop_cloth = image * cloth + (1 - cloth)
    off_cloth = image * (1-cloth) + cloth
    crop_arms = image * arms + (1-arms)
    crop_pants = image * pants + (1-pants)

    with open(path_pose_pkl, 'rb') as f:
        pose_label = pickle.load(f)
    pose_data = pose_label

    if (opt.version == 'viton'):
      shape = neckmake.fullmake(shape,pose_data) 
      shape = torch.from_numpy(shape)

    else:
        pose_key = [k for k in pose_data]
        if (16 not in pose_key) or (17 not in pose_key) or (2 not in pose_key) or (5 not in pose_key) or (3 not in pose_key) or (6 not in pose_key):
            shape = shape
            shape = torch.from_numpy(shape)
        else:
            shape = neckmake.fullmake(shape, pose_data)
            shape = torch.from_numpy(shape)

    point_num = 18
    pose_map = torch.zeros(point_num, H, W)
    r = 5
    pose = Image.new('L', (W, H))
    pose_draw = ImageDraw.Draw(pose)
    # for i in range(point_num):
    for i in [0,1,2,3,4,5,6,7,8,11,14,15,16,17]:
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
        'cloth' : cloth_,
        'cloth_mask': cloth_mask,
        'image': image,  # source image
        'head': crop_head,# cropped head from source image
        'pose': pose_map,#pose map
        'shape': shape,
        'upper_mask': cloth,# cropped cloth mask
        'crop_cloth': crop_cloth,#cropped cloth
        'crop_cloth_mask' : cloth,#cropped cloth mask
        'name' : name,
        'off_cloth': off_cloth,#source image - cloth
        'parse' : parse,
        'pants_mask': pants,
        'crop_pants': crop_pants,
        'crop_arms': crop_arms,

    }
    return results
    



def main():
    opt = get_opt()
    PYRAMID_HEIGHT = opt.PYRAMID_HEIGHT

    model1 = M1.UNet(opt,22,5)
    #device = torch.device("cuda:2")
    #model1.to(device)
    model1 = nn.DataParallel(model1,output_device=1)
    model2 = M2.FlowNet(5,4,1)
    #device = torch.device("cuda:2")
    #model2.to(device)
    model2 = nn.DataParallel(model2,output_device=1)
    model3 = M3.UNet(opt,33,4)
    #device = torch.device("cuda:2")
    #model3.to(device)
    model3 = nn.DataParallel(model3,output_device=1)
    model_CN = CN.ClothNormalizer(depth=5,nc=2)

    load_checkpoint(model1,opt.stage1_model_pth)
    load_checkpoint(model2,opt.stage2_model_pth)
    load_checkpoint(model3,opt.stage3_model_pth)
    load_checkpoint(model_CN,opt.clothnormalize_model_pth)

    set_requires_grad(model1,False)
    set_requires_grad(model2,False)
    set_requires_grad(model3,False)
    set_requires_grad(model_CN,False)

    if opt.mode == "all":
        names = os.listdir(opt.dataroot)
    elif opt.mode == "one":
        names = [opt.name]
    else:
        print("--mode should be all or one")
        assert(1==0)
    
    for name in names:
        inputs = load_inputs(opt,name)
        cloth = batch_cuda(inputs["cloth"])
        cloth_mask = batch_cuda(inputs["cloth_mask"])
        target_mask_real = batch_cuda(inputs["crop_cloth_mask"])
        target_pose = batch_cuda(inputs["pose"])
        # shape = batch_cuda(inputs["shape"])
        if opt.version=="MVC":
            print(cloth.shape, cloth_mask.shape, target_pose.shape)
            target_mask = model1(cloth,cloth_mask,target_pose,opt.is_top)
        else:
            target_mask = model1(cloth,cloth_mask, none)
        target_mask = (target_mask > -0.9).type(torch.float32)

        if not os.path.exists(osp.join(opt.result, name)): os.makedirs(osp.join(opt.result,name))
        imsave(target_mask,osp.join(opt.result,name,"stage1.jpg"))

        if opt.version=="MVC":
            params = model_CN(cloth_mask,target_mask)
            grid1 = projection.projection_grid(params,cloth_mask.shape)
            grid2 = projection.projection_grid(params,cloth.shape)
        
        else:
            theta = model_CN(cloth_mask,target_mask)
            grid1 = Ftnl.affine_grid(theta, cloth_mask.shape)
            grid2 = Ftnl.affine_grid(theta, cloth.shape)
        cloth = Ftnl.grid_sample(cloth , grid2,padding_mode="border").detach()

        imsave(cloth,osp.join(opt.result,name,"CN.jpg"))
        

        if opt.version=="MVC":
            [F, warp_cloth, warp_mask] = model2(torch.cat([cloth, cloth_mask], 1), target_mask)
        else:
            [F, warp_cloth, warp_mask, result_first, weight_list] = model2(torch.cat([cloth, cloth_mask], 1), target_mask)


        imsave(warp_cloth,osp.join(opt.result,name,"stage2.jpg"))

        warped = warp_cloth
        answer = batch_cuda(inputs["image"])
        target_mask = target_mask.to(target_mask_real.device)
        masking = (target_mask + target_mask_real).clamp(0,1)
        off_cloth = (answer * (1-masking) + masking).detach()
        pose = batch_cuda(inputs['pose'])
        if (opt.version == "viton"):
            shape = batch_cuda(inputs['shape'])
            shape = shape.unsqueeze_(1)
        head = batch_cuda(inputs['head'])
        pants = batch_cuda(inputs['crop_pants'])
        if opt.version == "MVC":
            result = model3(cloth,off_cloth,pants,warped,target_mask,head,pose)
        else:
            result = model3(cloth,off_cloth,pose,warped,shape,head,pants)
        imsave(off_cloth,osp.join(opt.result,name,"off_cloth.jpg"))
        img = (result[0].clone()+1)*0.5 * 255
        img = img.transpose(0,1).transpose(1,2)
        img = img.cpu().clamp(0,255)
        img = img.detach().numpy().astype('uint8')
        Image.fromarray(img).save(osp.join(opt.result,name+".jpg"))



    







if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    #torch.cuda.set_device(2)
    main()
