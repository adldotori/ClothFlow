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

def conversion(array, erode, dilate):
    if not 'numpy' in str(type(array)):
        array = array.cpu().numpy()
    array = array.squeeze()

    kernel = np.ones((3,3), np.uint8)
    array = cv2.erode(array, kernel, iterations=erode)
    array = cv2.dilate(array, kernel, iterations=dilate)
    
    array = torch.from_numpy(array)
    array = array.unsqueeze_(0)
    # array = array.unsqueeze_(0)
    return array

def get_opt():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/home/fashionteam/dataset/body_face_2")
    parser.add_argument("--mode", default = "all") # mode = all | one
    parser.add_argument("--name", default = "raw_0") # valid if mode == one
    parser.add_argument("--version", default = "viton") # version = MVC | vitons
    parser.add_argument("--is_top", default = True) # valid if version == MVC
    parser.add_argument("--PYRAMID_HEIGHT", default = 5)
    parser.add_argument("--stage1_model_pth", default = "stage1/checkpoints/tops/checkpoint_2_0.pth")
    # parser.add_argument("--stage1_model_pth", default = "stage1/checkpoints/tops/checkpoint_7_5000.pth")
    parser.add_argument("--stage2_model_pth", default = "stage2/checkpoints/tops/checkpoint_tmp_1.pth")
    parser.add_argument("--stage3_model_pth", default = "stage3/checkpoints/train/tops/checkpoint_final_0.pth")
    parser.add_argument("--clothnormalize_model_pth", default = "stage2/checkpoints/CN/train/tops/Epoch:14_00466.pth")
    parser.add_argument("--result", default = "test_bod_face4")
    parser.add_argument("--height", default = 512)
    parser.add_argument("--width", default = 512)

    opt = parser.parse_args()
    return opt

def load_inputs(opt,name):
    H = opt.height
    W = opt.width
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform_1ch = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.5], [0.5])])

    clothes = os.listdir(osp.join(opt.dataroot, 'cloth'))
    num_cloth = 5
    data_path = osp.join(opt.dataroot,name)
    path_cloth = osp.join(opt.dataroot,'cloth', clothes[num_cloth])
    path_image = osp.join(data_path,"image_.jpg")
    path_cloth_mask = osp.join(opt.dataroot, 'cloth-mask', clothes[num_cloth])
    path_segment = osp.join(data_path,"segment_.png")
    path_pose_pkl = osp.join(data_path,"pose.pkl")
    # path_mask = osp.join(data_path,"image_mask.jpg")

    cloth = Image.open(path_cloth).convert('RGB')
    image = Image.open(path_image).convert('RGB')
    cloth_mask = Image.open(path_cloth_mask)
    # mask = Image.open(path_mask)

    c_mask_array = np.array(cloth_mask)
    c_mask_array = (c_mask_array > 25).astype(np.float32)
    c_mask = torch.from_numpy(c_mask_array)
    cloth_mask = c_mask.unsqueeze_(0)
    # mask_array = np.array(mask)
    # mask = (mask_array > 0).astype(np.float32)
    # mask = (mask_array > 25).astype(np.float32)
    # mask = torch.from_numpy(mask)
    # mask = mask.unsqueeze_(0)

    cloth_ = transform(cloth)
    image = transform(image)

    seg = Image.open(path_segment)
    parse_array = np.array(seg)
    
    shape = (parse_array > 0).astype(np.float32)  # condition body shape
    background = (parse_array == 0).astype(np.float32)
    hair = (parse_array == 2).astype(np.float32)
    face = (parse_array == 13).astype(np.float32)
    hat = (parse_array == 1).astype(np.float32)
    tops = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32) + \
                (parse_array == 15).astype(np.float32) + \
                (parse_array == 14).astype(np.float32) + \
                (parse_array == 10).astype(np.float32)  
    bottoms = (parse_array == 8).astype(np.float32) + \
                (parse_array == 9).astype(np.float32) + \
                (parse_array == 12).astype(np.float32) + \
                (parse_array == 16).astype(np.float32) + \
                (parse_array == 17).astype(np.float32) + \
                (parse_array == 18).astype(np.float32) + \
                (parse_array == 19).astype(np.float32)
    cloth = (parse_array == 5).astype(np.float32) + \
                (parse_array == 6).astype(np.float32) + \
                (parse_array == 7).astype(np.float32)

    background = torch.from_numpy(background)
    hair = torch.from_numpy(hair)
    face = torch.from_numpy(face)
    hat = torch.from_numpy(hat)
    tops = torch.from_numpy(tops)
    bottoms = torch.from_numpy(bottoms)
    cloth = torch.from_numpy(cloth)

    seg_num = 3
    seg_map = torch.zeros(seg_num, H, W)
    seg_map[0] = hair
    seg_map[1] = face
    seg_map[2] = bottoms
    seg_map = seg_map.type(torch.float32)

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
    
    # print(head.shape)
    # print(np.mean(head))
    # head = conversion(head,4,2)
    # print(np.mean(head))
    # print(head.shape)
    head = torch.from_numpy(head)
    cloth = torch.from_numpy(cloth)
    arms = torch.from_numpy(arms)
    pants = torch.from_numpy(pants)
    shape = torch.from_numpy(shape)

    image = image
    # off_cloth_mask = shape - head - pants
    # off_cloth_mask[off_cloth_mask<0] = 0
    crop_head = image * head + (1 - head)
    crop_cloth = image * cloth + (1 - cloth)
    crop_arms = image * arms + (1-arms)
    crop_pants = (image * pants + (1-pants))

    remain = head + pants
    remain[remain>0]=1
    off_cloth = image * remain + (1 - remain) * 0
    with open(path_pose_pkl, 'rb') as f:
        pose_label = pickle.load(f)
    pose_data = pose_label
    
    # mask = neckmake.fullmake(mask, pose_data)
    pose_key = [k for k in pose_data] 
    
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

        if pointx > 1 and pointy > 1:
            draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
            pose_draw.rectangle((pointx - r, pointy - r, pointx + r, pointy + r), 'white', 'white')
        one_map = transform_1ch(one_map)
        pose_map[i] = one_map[0]
    
    results = {
        'cloth' : cloth_,
        'cloth_mask': cloth_mask,
        'image': image,  # source image
        'head': head,# cropped head from source image
        'pose': pose_map,#pose map
        'shape': shape,
        'upper_mask': cloth,# cropped cloth mask
        'crop_cloth': crop_cloth,#cropped cloth
        'crop_cloth_mask' : cloth,#cropped cloth mask
        'name' : name,
        'off_cloth': off_cloth,#source image - cloth
        'pants': pants,
        'crop_pants': crop_pants,
        'crop_arms': crop_arms,
        'arms': arms,
        'seg': seg_map,
        'crop_head': crop_head,
    }
    return results
    



def main():
    opt = get_opt()
    PYRAMID_HEIGHT = opt.PYRAMID_HEIGHT

    model1 = M1.UNet(opt,22,5)
    #device = torch.device("cuda:2")
    #model1.to(device)
    model1 = nn.DataParallel(model1)
    model1.eval()
    model2 = M2.FlowNet(5,4,1)
    model2.train()
    #device = torch.device("cuda:2")
    #model2.to(device)
    model2 = nn.DataParallel(model2)
    model3 = M3.UNet(opt,27,5)
    model3.eval()
    #device = torch.device("cuda:2")
    #model3.to(device)
    model3 = nn.DataParallel(model3)
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
        names = [i for i in os.listdir(opt.dataroot) if i[0] != 'c']
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
        # mask = batch_cuda(inputs['mask'])
        answer = batch_cuda(inputs["image"])
        arms = batch_cuda(inputs['arms'])
        head = batch_cuda(inputs['head'])
        pants = batch_cuda(inputs['pants'])
        seg = batch_cuda(inputs['seg'])
        shape = batch_cuda(inputs['shape'])
        off_cloth = batch_cuda(inputs['off_cloth'])

        ori_cloth = cloth

        # mask = mask.cpu().numpy()
        # mask = mask.squeeze()

        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=4)
        # mask = cv2.dilate(mask, kernel, iterations=2)
        
        # mask = torch.from_numpy(mask).cuda()
        # mask = mask.unsqueeze_(0)
        # mask = mask.unsqueeze_(0)

        # target_mask = model1(target_pose, cloth, cloth_mask, mask,  opt.is_top)
        target_mask = model1(target_pose, seg, cloth_mask , opt.is_top)
        target_mask = (target_mask > -0.9).type(torch.float32)
        
        target_mask = target_mask.cpu().numpy()
        target_mask = target_mask.squeeze()

        kernel = np.ones((3,3), np.uint8)
        target_mask = cv2.erode(target_mask, kernel, iterations=4)
        target_mask = cv2.dilate(target_mask, kernel, iterations=2)
        
        target_mask = torch.from_numpy(target_mask).cuda()
        target_mask = target_mask.unsqueeze_(0)
        target_mask = target_mask.unsqueeze_(0)

        # target_mask = target_mask * mask
        if not os.path.exists(osp.join(opt.result, name)): os.makedirs(osp.join(opt.result,name))
        masksave(target_mask,osp.join(opt.result,name,"stage1.jpg"))
        imsave(cloth, osp.join(opt.result, name, "cloth.jpg"))
        masksave(cloth_mask, osp.join(opt.result, name, "cloth_mask.jpg"))
        
        masksave(target_mask_real.view(target_mask_real.shape[0], 1, target_mask_real.shape[1], target_mask_real.shape[2]), osp.join(opt.result, name, "target_mask.jpg"))

        params = model_CN(cloth_mask,target_mask)
        grid1 = projection.projection_grid(params,cloth_mask.shape)
        grid2 = projection.projection_grid(params,cloth.shape)
        cloth_mask = Ftnl.grid_sample(cloth_mask , grid1).detach()
        cloth = Ftnl.grid_sample(cloth , grid2,padding_mode="border").detach()
        cloth = cloth * cloth_mask + (1 - cloth_mask)
        imsave(cloth,osp.join(opt.result,name,"CN.jpg"))

        [F, warp_cloth, warp_mask] = model2(torch.cat([cloth, cloth_mask], 1), target_mask)
        masksave(warp_mask, osp.join(opt.result, name, "warp_mask.jpg"))
        
        warp_cloth = warp_cloth * warp_mask + (1 - warp_mask)
        imsave(warp_cloth,osp.join(opt.result,name,"stage2.jpg"))

        answer = answer * shape + (1 - shape) * 0
        off_mask = target_mask_real + arms + warp_mask
        off_mask[off_mask > 1] = 1
        off_cloth = answer * (1- off_mask) + off_mask

        # remain = head + pants
        # remain[remain>0] = 1
        # arms_mask = torch.where(warp_mask>0, torch.zeros(1).cuda(), arms)
        # image_arms = answer * arms + (1 - arms) * 0
        # off_cloth = answer * remain + (1 - remain) * 0 + image_arms + cloth_mask

        imsave(off_cloth,osp.join(opt.result,name,"off_cloth.jpg"))
        imsave(answer, osp.join(opt.result, name, "image.jpg"))
        # masksave(mask, osp.join(opt.result, name, "body_mask.jpg"))
        
        warped = warp_cloth
        pose = batch_cuda(inputs['pose'])
        head = batch_cuda(inputs['head'])

        crop_head = batch_cuda(inputs['crop_head'])
        # pants = batch_cuda(inputs['crop_pants'])

        # imsave(head, osp.join(opt.result, name,"head.jpg"))
        # imsave(pants, osp.join(opt.result, name, "pants.jpg"))

        result = model3(pose,warped,off_cloth,crop_head)

        all_mask = target_mask + shape
        all_mask[all_mask > 0] = 1
        result = result * all_mask + (1 - all_mask) * 0
        result = torch.where(head>0, crop_head, result)
        imsave(result, osp.join(opt.result,name,"result.jpg"))

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    main()
